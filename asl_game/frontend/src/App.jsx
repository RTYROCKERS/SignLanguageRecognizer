import React, { useRef, useState, useEffect } from "react";
import words from "./words";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [started, setStarted] = useState(false);
  const [mediaPipeLoaded, setMediaPipeLoaded] = useState(false);
  const [currentWord, setCurrentWord] = useState(null);
  const [wordList, setWordList] = useState([]);
  const [index, setIndex] = useState(0);
  const [timer, setTimer] = useState(0);
  const [matched, setMatched] = useState(false);
  const intervalRef = useRef(null);
  const [triggerStart, setTriggerStart] = useState(false);

  // Create a ref to hold the latest currentWord
  const currentWordRef = useRef(currentWord);
  const indexRef = useRef(0);
  const wordListRef = useRef([]);
  const timerRef = useRef(0);
  const processingRef = useRef(false);
  useEffect(() => {
    indexRef.current = index;
  }, [index]);
  
  useEffect(() => {
    wordListRef.current = wordList;
  }, [wordList]);
  
  useEffect(() => {
    timerRef.current = timer;
  }, [timer]);
  
  // Update the ref whenever currentWord changes:
  useEffect(() => {
    currentWordRef.current = currentWord;
    console.log("Updated currentWord:", currentWord);
  }, [currentWord]);

  useEffect(() => {
    const checkMediaPipeReady = () => {
      if (
        window.Hands &&
        window.drawConnectors &&
        window.drawLandmarks &&
        window.HAND_CONNECTIONS &&
        window.Camera
      ) {
        setMediaPipeLoaded(true);
      } else {
        setTimeout(checkMediaPipeReady, 200);
      }
    };
    checkMediaPipeReady();
  }, []);

  const handleStart = () => {
    if (!mediaPipeLoaded) {
      alert("MediaPipe is still loading...");
      return;
    }
    setStarted(true);
    setTriggerStart(true);
  };

  useEffect(() => {
    if (triggerStart && videoRef.current) {
      
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      const hands = new window.Hands({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      hands.onResults(async (results) => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.scale(-1, 1); // Flip horizontally
        ctx.translate(-canvas.width, 0); // Move drawing back into visible area
        
        if (results.multiHandLandmarks?.length > 0) {
          const landmarks = results.multiHandLandmarks[0];
          window.drawConnectors(ctx, landmarks, window.HAND_CONNECTIONS, {
            color: "#00FF00",
            lineWidth: 2,
          });
          window.drawLandmarks(ctx, landmarks, {
            color: "#FF0000",
            lineWidth: 1,
          });

          // Use the ref to get the latest currentWord
          if (!currentWordRef.current) {
            console.log("Waiting for currentWord to be set...");
            return;
          }
          if (processingRef.current) {
            ctx.restore();
            return;
          }
          try {
            const res = await fetch("http://127.0.0.1:5000/predict", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ landmarks }),
            });
            const data = await res.json();
            //console.log("Current word inside onResults:", currentWordRef.current);
      
            if (
              currentWordRef.current &&
              data.prediction.toLowerCase() === currentWordRef.current.toLowerCase()
            ) {
              processingRef.current = true;
              setMatched(true);
              setTimeout(() => {
                console.log("word list length", wordListRef.current.length);
                console.log("index", indexRef.current);
                if (indexRef.current + 1 < wordListRef.current.length) {
                  setIndex((prevIndex) => prevIndex + 1);
                  setCurrentWord(wordListRef.current[indexRef.current + 1]);
                  setMatched(false);
                } else {
                  clearInterval(intervalRef.current);
                  alert(`🎉 Game Over! Time: ${timerRef.current}s`);
                  setStarted(false);
                }
                processingRef.current = false;
              }, 1000);
            }
          } catch (e) {
            console.error("Prediction error:", e);
          }
        }
        ctx.restore();
      });

      const camera = new window.Camera(video, {
        onFrame: async () => {
          await hands.send({ image: video });
        },
        width: 640,
        height: 480,
      });

      camera.start();

      if (wordList.length === 0) {
        const selected = [...words].sort(() => 0.5 - Math.random()).slice(0, 5);
        setWordList(selected);
        setCurrentWord(selected[0]);
        console.log("Selected first word:", selected[0]);
        setIndex(0);
        setTimer(0);
        intervalRef.current = setInterval(() => {
          setTimer((prev) => prev + 1);
        }, 1000);
      }
      setTriggerStart(false);
    }
  }, [triggerStart, videoRef, currentWord, wordList, index, timer]);

  return (
    <div
      className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-6 space-y-6"
      style={{
        backgroundImage: `url('/12.jpg')`,  // Background image URL
        backgroundSize: 'cover',  // Ensures the background image covers the full screen
        backgroundPosition: 'center',  // Centers the background image
        backgroundRepeat: 'no-repeat',  // Prevents repeating the background image
        height: '100vh',  // Full viewport height
        width: '100vw',  // Full viewport width
        position: 'relative',  // Important: Ensure the children are positioned relative to this parent
      }}
    >
      <h1 className="text-4xl font-extrabold text-yellow-300 tracking-wide">
        ✋ Sign Language Game
      </h1>

      {!started ? (
        <button
          onClick={handleStart}
          disabled={!mediaPipeLoaded}
          className={`px-6 py-3 text-lg rounded-md font-semibold transition-all duration-300 shadow-md ${
            mediaPipeLoaded
              ? 'bg-green-600 hover:bg-green-700 text-white'
              : 'bg-gray-500 text-gray-300 cursor-not-allowed'
          }`}
        >
          {mediaPipeLoaded ? '🎮 Start Game' : '⏳ Loading...'}
        </button>
      ) : (
        <>
          <div className="text-2xl">
            Current Word:{' '}
            <span className="font-bold text-pink-400 underline">{currentWord}</span>
          </div>
          <div className="text-lg text-blue-300">⏱️ Time: {timer}s</div>
          {matched && (
            <div className="text-green-400 text-lg font-semibold">✅ Matched!</div>
          )}

          {/* Canvas box with wood background */}
          <div
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 p-3 border-4 border-yellow-800 rounded-xl shadow-2xl"
            style={{
              backgroundImage: `url('/13.avif')`,
              backgroundRepeat: 'repeat',
              backgroundSize: 'cover',
              width: '640px',  // Ensure the canvas container has a defined width
              height: '480px',  // Ensure the canvas container has a defined height
            }}
          >
            <video ref={videoRef} style={{ display: 'none' }} width="640" height="480" />
            <canvas
              ref={canvasRef}
              width="640"
              height="480"
              className="rounded-md border-4 border-white bg-black"
            />
          </div>
        </>
      )}
    </div>

  );  
}
