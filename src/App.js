import { useRef, useEffect, useState } from 'react';
import * as faceapi from 'face-api.js';
import './App.css';

function App() {
  const idCardRef = useRef();
  const selfieRef = useRef();
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  useEffect(() => {
    const loadModels = async () => {
      await faceapi.nets.ssdMobilenetv1.loadFromUri('modal/weights');
      await faceapi.nets.faceLandmark68Net.loadFromUri('modal/weights');
      await faceapi.nets.faceRecognitionNet.loadFromUri('modal/weights');
      setIsModelLoaded(true); 
    };

    loadModels();
  }, []);

  const detectFaces = async () => {
    const idCardImage = idCardRef.current;
    const selfieImage = selfieRef.current;

    if (!idCardImage || !selfieImage) return;

   
    const idCardDetection = await faceapi
      .detectSingleFace(idCardImage, new faceapi.SsdMobilenetv1Options())
      .withFaceLandmarks()
      .withFaceDescriptor();

    const selfieDetection = await faceapi
      .detectSingleFace(selfieImage, new faceapi.SsdMobilenetv1Options())
      .withFaceLandmarks()
      .withFaceDescriptor();

 
    if (idCardDetection && selfieDetection) {
      const distance = faceapi.euclideanDistance(
        idCardDetection.descriptor,
        selfieDetection.descriptor
      );

    
      if (distance < 0.6) { 
        alert("Match found! Verification successful.");
      } else {
        alert("Faces do not match. Please try again.");
      }
    } else {
      alert("Face not detected in one or both images.");
    }
  };

  return (
    <div className="App">
      <div className="gallery">
        <img ref={idCardRef} src={require('./images/selfie.jpg')} alt="ID card" height="auto" />
      </div>

      <div className="gallery">
        <img ref={selfieRef} src={require('./images/selfie.jpg')} alt="Selfie" height="auto" />
      </div>

     
      <button onClick={detectFaces} disabled={!isModelLoaded}>
        {isModelLoaded ? 'Show Result' : 'Loading Models...'}
      </button>
    </div>
  );
}

export default App;
