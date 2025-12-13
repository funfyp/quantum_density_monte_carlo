Unity scaffold
===============

This folder contains a minimal scaffold for porting the simulation into Unity.

Files:
- QuantumSim.cs: a MonoBehaviour scaffold with placeholder methods and notes
  on porting the simulation from Python to C#.

Getting started
----------------
1. Create a new Unity project (recommended: Unity 2020 LTS or newer).
2. Copy the viz/unity folder into your project's Assets/ folder (e.g.
   Assets/QuantumDensityDemo/).
3. Open the scene, create an empty GameObject and attach the QuantumSim
   component. Configure gridSize and other public parameters in the Inspector.
4. Implement the simulation logic in RunSimulation() or port helper classes
   as needed.

Porting notes
--------------
- Numpy arrays (ndarray) should be converted to C# arrays (float[,]) or
  native arrays (NativeArray) for performance.
- For heavy numeric workloads, use the Unity Job System + Burst compiler to
  parallelize loops and get C-like performance.
- For visualization, convert grids to Texture2D (see GridToTexture) and
  display them on UI RawImage or on meshes.
- If the simulation relies on specialized libraries (FFT, linear algebra),
  consider Math.NET Numerics, native plugins, or reimplementing small
  pieces in C#.

Build targets
--------------
- The scaffold is platform-agnostic. Choose build targets in Unity (PC/Mac,
  WebGL, Android, iOS) depending on your needs. WebGL builds may require
  additional optimization and care around threading (Jobs/Burst support
  differs by platform).

License / Attribution
----------------------
Refer to the main project LICENSE for reuse of code and assets. This scaffold
is intended as a starting point for integration.
