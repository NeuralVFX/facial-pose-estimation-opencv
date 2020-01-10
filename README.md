![](examples/unity_example.gif)
# Facial-Pose-Estimation-OpenCV

This is a `C++` implimentation of `Realtime Facial and Headpose Estimation` using `OpenCV`, `DLIB` and a `CNN` trained in `Pytorch`.

## About
- The output of the Visual Studio Project is a `DLL`
- The DLL outputs the `Transform Matrix` of the head, a set of `Blendshape Values`, and the pixels of the image
- An example of using this `DLL` with `Unity` can be found in this project: [facial-pose-estimation-unity](https://github.com/NeuralVFX/facial-pose-estimation-unity)
- Utilizes a Neural Net from thie project: [facial-pose-estimation-pytorch](https://github.com/NeuralVFX/facial-pose-estimation-pytorch)
- This Runs on a live video stream

## Estimation Pipeline Diagram
![](examples/pipeline_b.png)

## Code Usage
Usage instructions found here: [user manual page](USAGE.md).




