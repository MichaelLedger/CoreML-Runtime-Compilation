# ImageScore Swift Playground App

This Swift Playground App demonstrates **CoreML Runtime Compilation** with automatic model download, extraction, compilation, and inference.

## Features

✅ **Download CoreML Model** from remote server  
✅ **Extract ZIP** using [ZipArchive](https://github.com/ZipArchive/ZipArchive) SPM package  
✅ **Compile `.mlpackage`** to `.mlmodelc` at runtime  
✅ **Run Predictions** with UIQACoreMLPredictor  
✅ **Beautiful SwiftUI Interface** with real-time logs and status  

## Setup

### 1. Add Test Image

You need to add a test image to the project:

1. Find or create a test image (e.g., `forest.jpg`)
2. Drag it into the Xcode project
3. Make sure it's added to the target
4. The code looks for an image named `"forest"` in assets

### 2. Dependencies

The project uses **ZipArchive** via Swift Package Manager. It's already configured in `Package.swift`:

```swift
dependencies: [
    .package(url: "git@github.com:ZipArchive/ZipArchive.git", "2.6.0"..<"3.0.0")
]
```

### 3. Network Configuration

The model downloads from:
```
http://10.4.2.8:8080/view/PB_iOS/job/PB_BestPhoto_POC/ws/PhotoBooks/MyDealsSDK/mydeals/BestPhoto/UIQAModel.mlpackage.zip
```

Make sure:
- Your device/simulator can access this URL
- For iOS apps, add the domain to `Info.plist` under `NSAppTransportSecurity` if needed

## Usage

1. **Open the project** in Xcode or Swift Playgrounds
2. **Run the app** on iOS device or simulator (iOS 16.0+)
3. **Tap "Download, Compile & Predict"** button
4. Watch the logs as it:
   - Downloads the model zip file
   - Extracts the `.mlpackage`
   - Compiles it to `.mlmodelc`
   - Runs inference on the test image
5. See the **Quality Score** and **Inference Time** displayed

## How It Works

### Workflow

```
Download ZIP → Extract .mlpackage → Compile to .mlmodelc → Load Model → Run Prediction
```

### Components

- **ContentView.swift** - SwiftUI interface with download/compile/predict logic
- **UIQACoreMLPredictor.swift** - CoreML model wrapper with prediction logic
- **MDBestPhotoPredictor.swift** - Protocol for photo quality predictors
- **MDBestPhotoExtension.swift** - UIImage helper extensions

### Key Features

- **Caching**: Downloaded models are cached in Documents directory
- **Async/Await**: Modern Swift concurrency for smooth UI
- **Error Handling**: Comprehensive error logging
- **Playground Detection**: Uses CPU-only mode in playgrounds for compatibility

## Files Structure

```
ImageScore.swiftpm/
├── ContentView.swift              # Main SwiftUI view
├── MyApp.swift                    # App entry point
├── UIQACoreMLPredictor.swift      # CoreML predictor
├── MDBestPhotoPredictor.swift     # Protocol
├── MDBestPhotoExtension.swift     # Extensions
└── Package.swift                  # SPM configuration
```

## Troubleshooting

### Image not found
- Add a test image to the project assets
- Update the image name in `runPrediction()` if using a different name

### Download fails
- Check network connectivity
- Verify the URL is accessible
- Check iOS App Transport Security settings

### Compilation fails
- Ensure the `.mlpackage` structure is valid
- Check iOS version compatibility (16.0+)
- Try clearing cache and re-downloading

### Extraction fails
- Verify the ZIP file is valid
- Check file permissions
- Try using the "Clear Model Cache" button

## Credits

- **ZipArchive** by [ZipArchive/ZipArchive](https://github.com/ZipArchive/ZipArchive)
- **CoreML** by Apple

## License

MIT License

