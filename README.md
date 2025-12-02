# CoreML-Runtime-Compilation

A demo of CoreML runtime compilation. 

This includes:
* Downloading the ML Model from the server dynamically 
* compiling it into .mlmodelc 
* using custom Model Input file to make predictions. 

## Playground logs below:
```
Initialized
Destination directory: /Users/xxx/Library/Developer/XCPGDevices/706D1A66-3EC6-4A78-BCD8-E060BCAFA013/data/Containers/Data/Application/05FD7594-792D-4B9E-B1B3-29700480B069/Documents/Shared Playground Data
✓ Created destination directory
Model file not found, starting download...
HTTP Status Code: 200
Download completed. Temporary file at: /Users/xxx/Library/Developer/XCPGDevices/706D1A66-3EC6-4A78-BCD8-E060BCAFA013/data/Containers/Data/Application/05FD7594-792D-4B9E-B1B3-29700480B069/tmp/CFNetworkDownload_zS2UaM.tmp
Copying from /Users/xxx/Library/Developer/XCPGDevices/706D1A66-3EC6-4A78-BCD8-E060BCAFA013/data/Containers/Data/Application/05FD7594-792D-4B9E-B1B3-29700480B069/tmp/CFNetworkDownload_zS2UaM.tmp to /Users/xxx/Library/Developer/XCPGDevices/706D1A66-3EC6-4A78-BCD8-E060BCAFA013/data/Containers/Data/Application/05FD7594-792D-4B9E-B1B3-29700480B069/Documents/Shared Playground Data/GoogleNetPlaces.mlmodel
✓ Model downloaded and saved successfully
Downloaded file size: 24754375 bytes
Compiling model at: /Users/xxx/Library/Developer/XCPGDevices/706D1A66-3EC6-4A78-BCD8-E060BCAFA013/data/Containers/Data/Application/05FD7594-792D-4B9E-B1B3-29700480B069/Documents/Shared Playground Data/GoogleNetPlaces.mlmodel
✓ Model compiled successfully to: /Users/xxx/Library/Developer/XCPGDevices/706D1A66-3EC6-4A78-BCD8-E060BCAFA013/data/Containers/Data/Application/05FD7594-792D-4B9E-B1B3-29700480B069/tmp/GoogleNetPlaces.mlmodelc
compiledURL:file:///Users/xxx/Library/Developer/XCPGDevices/706D1A66-3EC6-4A78-BCD8-E060BCAFA013/data/Containers/Data/Application/05FD7594-792D-4B9E-B1B3-29700480B069/tmp/GoogleNetPlaces.mlmodelc
Loading model from: /Users/xxx/Library/Developer/XCPGDevices/706D1A66-3EC6-4A78-BCD8-E060BCAFA013/data/Containers/Data/Application/05FD7594-792D-4B9E-B1B3-29700480B069/tmp/GoogleNetPlaces.mlmodelc
✓ Model loaded successfully
✓ Image converted to pixel buffer
Making prediction...
✓ Prediction result: forest_path
```

## Image Score Playground


```
/// Construct a model given the location of its on-disk representation. Returns nil on error.
@available(iOS 12.0, *)
public convenience init(contentsOf url: URL, configuration: MLModelConfiguration) throws
```

**NOTE: `ImageScore.swiftpm` should be opened with Xcode, not Swift Playground which do not support Swift Package Manager(SPM) for now.**

```
[BestPhoto] 2025-12-02 03:43:20 +0000 == 🎮 Running in Playground - Using CPU only for compatibility
[BestPhoto] 2025-12-02 03:43:20 +0000 == ⚙️ Model configuration set: computeUnits = 0
[BestPhoto] 2025-12-02 03:43:20 +0000 == 💾 Available memory: 7681 MB
[BestPhoto] 2025-12-02 03:43:20 +0000 == 🔄 Attempting to load model with computeUnits: 0
[BestPhoto] 2025-12-02 03:45:20 +0000 == ⏱ Model loading timed out after 120.00s
[BestPhoto] 2025-12-02 03:45:20 +0000 == 🔄 Retrying with CPU only as fallback...
[BestPhoto] 2025-12-02 03:45:20 +0000 == 💻 Loading model with CPU only...
[BestPhoto] 2025-12-02 03:45:20 +0000 == ✅ Model loaded successfully with CPU only (total time: 120.59s)
[BestPhoto] 2025-12-02 03:45:20 +0000 == ✅ Successfully loaded UIQA Core ML model (took 120.59s)
[BestPhoto] 2025-12-02 03:45:20 +0000 == ✅ Model successfully initialized and cached in actor
```

**NOTE: Runtime loading must running in main thread!!!**

```
[BestPhoto] 2025-12-02 03:55:02 +0000 == 🔍 Device Model: iPhone17,3
[BestPhoto] 2025-12-02 03:55:02 +0000 == 🔍 iOS: 26.2.0
[BestPhoto] 2025-12-02 03:55:02 +0000 == 🔍 Neural Engine: ✅ Available
[BestPhoto] 2025-12-02 03:55:02 +0000 == 🧠 Using CPU + Neural Engine (iOS 18+ with ANE)
[BestPhoto] 2025-12-02 03:55:02 +0000 == ⚙️ Model configuration set: computeUnits = 3
[BestPhoto] 2025-12-02 03:55:02 +0000 == 💾 Available memory: 7681 MB
[BestPhoto] 2025-12-02 03:55:02 +0000 == 🔄 Attempting to load model with computeUnits: 3
Loading model from path:file:///private/var/mobile/Containers/Data/Application/1EEF73A5-FEB0-4966-A67B-DE6B4EB0F9FD/tmp/UIQAModel_4A09E0DA-5CBD-4BAE-AF07-A501789612C4.mlmodelc
[BestPhoto] 2025-12-02 03:55:04 +0000 == 🎯 Model loaded successfully
[BestPhoto] 2025-12-02 03:55:04 +0000 == ✅ Successfully loaded UIQA Core ML model (took 2.01s)
[BestPhoto] 2025-12-02 03:55:04 +0000 == ✅ Model successfully initialized and cached in actor
```

## Terminal logs below:
```
📸 Loading selected photo...
✅ Photo loaded successfully
   Size: 960x2079

🔮 Running prediction...
✓ Using custom selected image
[BestPhoto] 2025-12-02 03:56:11 +0000 == ♻️ Returning cached model from actor
🎉 Prediction successful!
   Score: 0.72558594
   Inference Time: 0.1231s
```

## For full tutorial, visit this link. 

https://hadiajalil.com/coreml-compilingmodel/
