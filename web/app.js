/* ==========================================
   DONUTS WEB DASHBOARD
   Camera + WebSocket AI Streaming
========================================== */


/* ==========================================
   DOM ELEMENTS
========================================== */

const camera =
    document.getElementById("camera");

const aiCanvas =
    document.getElementById("aiCanvas");

const cameraPlaceholder =
    document.getElementById("cameraPlaceholder");

const startCamera =
    document.getElementById("startCamera");

const resetButton =
    document.getElementById("resetButton");

const activity =
    document.getElementById("activity");

const nextStep =
    document.getElementById("nextStep");

const message =
    document.getElementById("message");

const connectionDot =
    document.getElementById("connectionDot");

const connectionText =
    document.getElementById("connectionText");

const experimentStatus =
    document.getElementById("experimentStatus");

const stepElements = [
    document.getElementById("step1"),
    document.getElementById("step2"),
    document.getElementById("step3"),
    document.getElementById("step4")
];


/* ==========================================
   STATE
========================================== */

let cameraStream = null;

let websocket = null;

let captureCanvas = null;
let captureContext = null;

let sendingFrame = false;

let aiRunning = false;

let frameCounter = 0;


/*
   Expose WebSocket globally so Chrome
   DevTools Console can inspect it.

   Example:
   donutsWebSocket.readyState
*/
window.donutsWebSocket = null;


/* ==========================================
   CAMERA
========================================== */

async function startWebcam() {

    try {

        console.log(
            ">>> DONUTS: Requesting camera..."
        );

        message.textContent =
            "Requesting camera access...";


        /* ----------------------------------
           Request webcam
        ---------------------------------- */

        cameraStream =
            await navigator.mediaDevices.getUserMedia({

                video: {

                    width: {
                        ideal: 640
                    },

                    height: {
                        ideal: 480
                    },

                    facingMode: "user"
                },

                audio: false
            });


        /* ----------------------------------
           Attach camera stream
        ---------------------------------- */

        camera.srcObject =
            cameraStream;

        await camera.play();


        console.log(
            ">>> DONUTS: Camera connected"
        );


        /* ----------------------------------
           UI
        ---------------------------------- */

        cameraPlaceholder.style.display =
            "none";

        camera.style.display =
            "none";

        aiCanvas.style.display =
            "block";


        /* ----------------------------------
           Capture canvas
        ---------------------------------- */

        captureCanvas =
            document.createElement("canvas");

        captureCanvas.width = 640;
        captureCanvas.height = 360;

        captureContext =
            captureCanvas.getContext("2d");


        /* ----------------------------------
           AI display canvas
        ---------------------------------- */

        aiCanvas.width = 640;
        aiCanvas.height = 360;


        message.textContent =
            "Camera connected. Connecting to DONUTS AI...";


        /* ----------------------------------
           Connect AI
        ---------------------------------- */

        connectAI();

    }

    catch (error) {

        console.error(
            ">>> DONUTS CAMERA ERROR:",
            error
        );

        message.textContent =
            "Camera access failed. Please allow webcam access in Chrome.";

    }

}


/* ==========================================
   WEBSOCKET
========================================== */

function connectAI() {

    /* --------------------------------------
       Prevent duplicate connections
    -------------------------------------- */

    if (
        websocket &&
        (
            websocket.readyState === WebSocket.OPEN ||
            websocket.readyState === WebSocket.CONNECTING
        )
    ) {

        console.log(
            ">>> DONUTS: WebSocket already connected/connecting"
        );

        return;
    }


    /* --------------------------------------
       Select protocol
    -------------------------------------- */

    const protocol =
        window.location.protocol === "https:"
            ? "wss:"
            : "ws:";


    const websocketURL =
        `${protocol}//${window.location.host}/ws/ai`;


    console.log(
        "=========================================="
    );

    console.log(
        ">>> DONUTS: Connecting AI WebSocket"
    );

    console.log(
        ">>> URL:",
        websocketURL
    );


    /* --------------------------------------
       Create WebSocket
    -------------------------------------- */

    websocket =
        new WebSocket(
            websocketURL
        );


    /*
       Make available in DevTools.
    */

    window.donutsWebSocket =
        websocket;


    websocket.binaryType =
        "arraybuffer";


    /* ======================================
       ON OPEN
    ====================================== */

    websocket.onopen =
        function () {

            console.log(
                "=========================================="
            );

            console.log(
                ">>> DONUTS AI WEBSOCKET CONNECTED"
            );

            console.log(
                ">>> READY STATE:",
                websocket.readyState
            );

            console.log(
                "=========================================="
            );


            connectionDot.style.background =
                "#36d399";

            connectionText.textContent =
                "AI CONNECTED";

            message.textContent =
                "DONUTS AI connected. Processing camera...";


            aiRunning = true;

            sendingFrame = false;

            frameCounter = 0;


            /*
               Send first frame immediately.
            */

            sendNextFrame();

        };


    /* ======================================
       ON MESSAGE
    ====================================== */

    websocket.onmessage =
        function (event) {

            console.log(
                ">>> DONUTS: AI response received"
            );


            /*
               Always release the frame lock
               when a response arrives.
            */

            sendingFrame = false;


            try {

                /* --------------------------
                   Parse response
                -------------------------- */

                const result =
                    JSON.parse(
                        event.data
                    );


                console.log(
                    ">>> DONUTS: Frame response parsed"
                );


                /* --------------------------
                   Update dashboard
                -------------------------- */

                try {

                    updateDashboard(
                        result
                    );

                }

                catch (error) {

                    console.error(
                        ">>> DONUTS DASHBOARD ERROR:",
                        error
                    );

                }


                /* --------------------------
                   Display processed frame
                -------------------------- */

                try {

                    displayAIFrame(
                        result.frame
                    );

                }

                catch (error) {

                    console.error(
                        ">>> DONUTS FRAME DISPLAY ERROR:",
                        error
                    );

                }


                /*
                   Continue processing.

                   Small delay prevents the browser
                   from immediately hammering the
                   Python server.
                */

                if (
                    aiRunning &&
                    websocket &&
                    websocket.readyState === WebSocket.OPEN
                ) {

                    setTimeout(
                        function () {

                            sendNextFrame();

                        },

                        50
                    );

                }

            }

            catch (error) {

                console.error(
                    ">>> DONUTS AI RESPONSE ERROR:",
                    error
                );


                /*
                   Even if JSON/dashboard processing
                   fails, try to continue the stream.
                */

                if (
                    aiRunning &&
                    websocket &&
                    websocket.readyState === WebSocket.OPEN
                ) {

                    setTimeout(
                        function () {

                            sendNextFrame();

                        },

                        100
                    );

                }

            }

        };


    /* ======================================
       ON ERROR
    ====================================== */

    websocket.onerror =
        function (error) {

            console.error(
                "=========================================="
            );

            console.error(
                ">>> DONUTS WEBSOCKET ERROR"
            );

            console.error(
                error
            );

            console.error(
                "=========================================="
            );


            connectionDot.style.background =
                "#e5484d";

            connectionText.textContent =
                "AI CONNECTION ERROR";

            message.textContent =
                "Unable to connect to DONUTS AI.";

        };


    /* ======================================
       ON CLOSE
    ====================================== */

    websocket.onclose =
        function (event) {

            console.log(
                "=========================================="
            );

            console.log(
                ">>> DONUTS AI WEBSOCKET CLOSED"
            );

            console.log(
                ">>> CLOSE CODE:",
                event.code
            );

            console.log(
                ">>> CLOSE REASON:",
                event.reason
            );

            console.log(
                ">>> CLEAN:",
                event.wasClean
            );

            console.log(
                "=========================================="
            );


            aiRunning = false;

            sendingFrame = false;


            connectionDot.style.background =
                "#e5484d";

            connectionText.textContent =
                "AI OFFLINE";


            /*
               Keep the object available for
               Chrome DevTools inspection.
            */

            window.donutsWebSocket =
                websocket;

        };

}


/* ==========================================
   SEND FRAME
========================================== */

function sendNextFrame() {

    /* --------------------------------------
       Basic checks
    -------------------------------------- */

    if (!aiRunning) {

        console.log(
            ">>> FRAME STOPPED: AI not running"
        );

        return;
    }


    if (!websocket) {

        console.log(
            ">>> FRAME STOPPED: No WebSocket"
        );

        return;
    }


    if (
        websocket.readyState !== WebSocket.OPEN
    ) {

        console.log(
            ">>> FRAME STOPPED: WebSocket state =",
            websocket.readyState
        );

        return;
    }


    /*
       Do not send another frame while Python
       is processing the previous frame.
    */

    if (sendingFrame) {

        console.log(
            ">>> FRAME SKIPPED: Previous frame still processing"
        );

        return;
    }


    /*
       Camera may not have dimensions yet.
    */

    if (
        !camera.videoWidth ||
        !camera.videoHeight
    ) {

        console.log(
            ">>> Camera dimensions not ready"
        );

        setTimeout(
            sendNextFrame,
            100
        );

        return;
    }


    /* --------------------------------------
       Lock frame
    -------------------------------------- */

    sendingFrame = true;

    frameCounter++;


    console.log(
        `>>> DONUTS: Sending frame #${frameCounter}`
    );


    /* --------------------------------------
       Draw webcam frame
    -------------------------------------- */

    captureContext.drawImage(

        camera,

        0,
        0,

        captureCanvas.width,
        captureCanvas.height
    );


    /* --------------------------------------
       Convert to JPEG
    -------------------------------------- */

    captureCanvas.toBlob(

        function (blob) {

            /*
               Blob creation failed.
            */

            if (!blob) {

                console.error(
                    ">>> DONUTS: Failed to create JPEG blob"
                );

                sendingFrame = false;

                setTimeout(
                    sendNextFrame,
                    100
                );

                return;
            }


            /*
               WebSocket may have closed while
               toBlob was running.
            */

            if (
                !websocket ||
                websocket.readyState !== WebSocket.OPEN
            ) {

                console.error(
                    ">>> DONUTS: WebSocket closed before frame send"
                );

                sendingFrame = false;

                return;
            }


            try {

                websocket.send(
                    blob
                );

                console.log(
                    `>>> DONUTS: Frame #${frameCounter} sent (${blob.size} bytes)`
                );

            }

            catch (error) {

                console.error(
                    ">>> DONUTS FRAME SEND ERROR:",
                    error
                );

                sendingFrame = false;

            }

        },

        "image/jpeg",

        0.65
    );

}


/* ==========================================
   DISPLAY AI FRAME
========================================== */

function displayAIFrame(
    base64Frame
) {

    if (!base64Frame) {

        console.warn(
            ">>> DONUTS: AI response has no frame"
        );

        return;
    }


    const image =
        new Image();


    image.onload =
        function () {

            const context =
                aiCanvas.getContext("2d");


            context.drawImage(

                image,

                0,
                0,

                aiCanvas.width,
                aiCanvas.height
            );

        };


    image.onerror =
        function (error) {

            console.error(
                ">>> DONUTS AI IMAGE ERROR:",
                error
            );

        };


    image.src =
        "data:image/jpeg;base64," +
        base64Frame;

}


/* ==========================================
   UPDATE DASHBOARD
========================================== */

function updateDashboard(
    result
) {

    /* --------------------------------------
       Current activity
    -------------------------------------- */

    if (result.activity) {

        activity.textContent =
            result.activity;

    }


    /* --------------------------------------
       Next step
    -------------------------------------- */

    if (result.next_step) {

        nextStep.textContent =
            result.next_step
                .replaceAll(
                    "_",
                    " "
                );

    }


    /* --------------------------------------
       Experiment status
    -------------------------------------- */

    if (result.completed) {

        experimentStatus.textContent =
            "COMPLETE";

        experimentStatus.className =
            "status";

        message.textContent =
            "Experiment completed successfully.";

    }

    else if (
        result.controller_status ===
        "WARNING"
    ) {

        experimentStatus.textContent =
            "WARNING";

        message.textContent =
            result.controller_message ||
            "Incorrect experiment sequence.";

    }

    else {

        experimentStatus.textContent =
            "RUNNING";

        message.textContent =
            result.controller_message ||
            "DONUTS is monitoring the experiment.";

    }


    /* --------------------------------------
       Experiment steps
    -------------------------------------- */

    const currentStep =
        result.step_index || 0;

    const totalSteps =
        result.total_steps || 4;


    stepElements.forEach(

        function (step, index) {

            if (!step) {
                return;
            }


            step.classList.remove(
                "active"
            );

            step.classList.remove(
                "completed"
            );


            if (
                index < currentStep
            ) {

                step.classList.add(
                    "completed"
                );

            }

            else if (
                index === currentStep &&
                currentStep < totalSteps
            ) {

                step.classList.add(
                    "active"
                );

            }

        }

    );


    /* --------------------------------------
       FPS
    -------------------------------------- */

    if (result.fps) {

        console.log(
            "DONUTS AI FPS:",
            result.fps
        );

    }

}


/* ==========================================
   RESET EXPERIMENT
========================================== */

function resetExperiment() {

    activity.textContent =
        "WAITING";

    nextStep.textContent =
        "APPROACH BOTTLE";

    experimentStatus.textContent =
        "READY";

    message.textContent =
        "Experiment reset.";

}


/* ==========================================
   BUTTON EVENTS
========================================== */

startCamera.addEventListener(

    "click",

    startWebcam

);


resetButton.addEventListener(

    "click",

    resetExperiment

);


/* ==========================================
   CLEANUP
========================================== */

window.addEventListener(

    "beforeunload",

    function () {

        console.log(
            ">>> DONUTS: Closing dashboard"
        );


        aiRunning = false;

        sendingFrame = false;


        if (websocket) {

            try {

                websocket.close();

            }

            catch (error) {

                console.error(
                    error
                );

            }

        }


        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(

                    track =>
                        track.stop()

                );

        }

    }

);


/* ==========================================
   INITIALIZATION
========================================== */

console.log(
    "=========================================="
);

console.log(
    "DONUTS WEB DASHBOARD JS READY"
);

console.log(
    ">>> Camera system ready"
);

console.log(
    ">>> WebSocket system ready"
);

console.log(
    ">>> Frame streaming system ready"
);

console.log(
    "=========================================="
);