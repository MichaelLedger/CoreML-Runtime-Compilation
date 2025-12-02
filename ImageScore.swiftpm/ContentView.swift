import SwiftUI
import CoreML
import ZipArchive
import PhotosUI

struct ContentView: View {
    @State private var statusMessage = "Ready to start"
    @State private var logs: [String] = []
    @State private var isProcessing = false
    @State private var predictionScore: Float?
    @State private var inferenceTime: Double?
    @State private var downloadProgress: Double = 0.0
    @State private var isDownloading: Bool = false
    @State private var compiledModelURL: URL?
    @State private var predictor: UIQACoreMLPredictor?
    @State private var selectedImage: UIImage?
    @State private var selectedPhotoItem: PhotosPickerItem?
    
    // Best Photo - Image Quality Assessment (iOS)
    // https://planetart.atlassian.net/wiki/x/BoCH9w
    
    // below links will timeout, return 403, access denied
    
    // Model URLs and paths
    //test full-weight model (UIQA_UHD_IQA_V2__NR_epoch_42_SRCC_0.726446.pth)
//    let modelDownloadURL = URL(string: "https://media-cdn.atlassian.com/us-west-2/v2/cdn/client/5addb755-59ac-4a85-9f77-3e4003d7eb6a/file/ae62b22c-5945-47b5-9d19-a1d29bff582c/binary?collection=contentId-4152852486&dl=true&max-age=2592000&token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImdlbmVyaWMta2V5cGFpci9kdC1hcGktZmlsZXN0b3JlL2Nkbi1hdXRoLS1xc3I3MWNtaXFsMWdncDBoIn0.eyJzdWIiOiI1YWRkYjc1NS01OWFjLTRhODUtOWY3Ny0zZTQwMDNkN2ViNmEiLCJjbGllbnRJZCI6IjVhZGRiNzU1LTU5YWMtNGE4NS05Zjc3LTNlNDAwM2Q3ZWI2YSIsImV4dElkIjoiZDkyZmU3YTgtMjk4ZC00YzIwLWFkYjAtYTRiZWUyODhkY2I3IiwiaW50SWQiOiJkOTJmZTdhOC0yOThkLTRjMjAtYWRiMC1hNGJlZTI4OGRjYjciLCJpc29sYXRlZCI6ZmFsc2UsInJlc291cmNlVHlwZSI6MywiaXBhIjpmYWxzZSwiZmIiOiJ0ZHAtb25seSIsImZtIjoidGRwLW9ubHkiLCJjbiI6InRkcC1vbmx5IiwicnMiOiJ0ZHAtb25seSIsImlzcyI6ImdlbmVyaWMta2V5cGFpci9kdC1hcGktZmlsZXN0b3JlIiwiYXVkIjoibWVkaWEiLCJpYXQiOjE3NjQ1Nzc5ODgsIm5iZiI6MTc2NDU3Nzk4OCwiZXhwIjoxNzY0NTgwNDUxLCJqdGkiOiI0ODgyOWRhNjg3MjFjY2ZkNzRiZGQwNWI4M2Y0ZjJiYTc5NDZiMWE3In0.kjFZM1pSVyLMNkNGQ9gwx4gPQsQwf8NxOXrsHXPUnpcG6nHZpHC1HhnoAg8V1-YGrdNJOeUdO58BNOHCd1YFDvLVAdRYLFcD84uGGvhaKICysPDigkBDCn2cNJWI6IxbwLjqeWdm9wLHLzGoE_bXx4FqgPfigc3HjZEM6T3GPI0hOZdFLu_mfJRhFZnMbffsf3VtgrZ_tYpM4uwjmWDmRZ8_yLOVwxRfUc6cVl25ajsRdWiJ5rwrAtrI1BsUxOypryiW_OJfaQH8HZUYnZcemR7It0Qx0Q0SfcfsVyRhuvtUPrfv3TjV6PRdsJFMMgwpMMVFFY_RxRqiPQAKdObjbw&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9tZWRpYS1jZG4uYXRsYXNzaWFuLmNvbS91cy13ZXN0LTIvdjIvY2RuL2NsaWVudC81YWRkYjc1NS01OWFjLTRhODUtOWY3Ny0zZTQwMDNkN2ViNmEvZmlsZS9hZTYyYjIyYy01OTQ1LTQ3YjUtOWQxOS1hMWQyOWJmZjU4MmMvYmluYXJ5PyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjQ1ODA0NTF9fX1dfQ__&Key-Pair-Id=K3CEQBLVYJW0FT&Signature=HrcagBqP3PUjn06w9qV9XpzrZCBNEC10Z3r3CD~P8kqPntKbAWOluk18Z8IaHfd3au9ZO1XysHzi9kGHbsx0NrvbihAJ0RotCaX69Hrye-7k~Wu9p94nxx-Z6qQ5FYQbUOBiaeKArFNQ2uMgYHvAXgWuTCkivAWPLmK0-Sz7trUhAqQMbuesjbRg21g1sn5PWmsdOeZwggyymYsrxqlxZV8vm3Pu1CF-ScYbuC0I9ECZt357o4466poVWD1oTgKnWtyFX1g52tgYDmck5o9u7Gn28MbR13Ko9o6F74FwrLkYF~ubK~qgsmaxupce0MedoIbV97~6N0PXnTtFTLf3Sg__")!
    
    //test light-weight model (UIQA_light_UHD_IQA_V2__NR_epoch_9_SRCC_0.466073.pth)
    let modelDownloadURL = URL(string: "https://media-cdn.atlassian.com/us-west-2/v2/cdn/client/5addb755-59ac-4a85-9f77-3e4003d7eb6a/file/7f5d5722-1842-4371-a6b1-da1f3c571e6e/binary?collection=contentId-4152852486&dl=true&max-age=2592000&token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImdlbmVyaWMta2V5cGFpci9kdC1hcGktZmlsZXN0b3JlL2Nkbi1hdXRoLS1xc3I3MWNtaXFsMWdncDBoIn0.eyJzdWIiOiI1YWRkYjc1NS01OWFjLTRhODUtOWY3Ny0zZTQwMDNkN2ViNmEiLCJjbGllbnRJZCI6IjVhZGRiNzU1LTU5YWMtNGE4NS05Zjc3LTNlNDAwM2Q3ZWI2YSIsImV4dElkIjoiZDkyZmU3YTgtMjk4ZC00YzIwLWFkYjAtYTRiZWUyODhkY2I3IiwiaW50SWQiOiJkOTJmZTdhOC0yOThkLTRjMjAtYWRiMC1hNGJlZTI4OGRjYjciLCJpc29sYXRlZCI6ZmFsc2UsInJlc291cmNlVHlwZSI6MywiaXBhIjpmYWxzZSwiZmIiOiJ0ZHAtb25seSIsImZtIjoidGRwLW9ubHkiLCJjbiI6InRkcC1vbmx5IiwicnMiOiJ0ZHAtb25seSIsImlzcyI6ImdlbmVyaWMta2V5cGFpci9kdC1hcGktZmlsZXN0b3JlIiwiYXVkIjoibWVkaWEiLCJpYXQiOjE3NjQ2NDYzMzgsIm5iZiI6MTc2NDY0NjMzOCwiZXhwIjoxNzY0NjQ3MjY3LCJqdGkiOiJhNzIwM2YwZDk3ZDEwMTE5OGY5YWNiZDBlOTY5ZWU2MTRiZjQ1YTE2In0.0j-7QkNzNN_8iFROqUMZTVAE-z_nwxXI6NAEIdrRxgixNcnzAXwVp5K5KPwccMYEQsQtybQ1syF5PrAKHiyufcoOyJd9ntk9Vskb_96I-_SOU1VYnfFNjoCyRXIlOauQt8qw4Fis2Rh62SmUkR5ZKXF1fLdFK5ExkDt6NhFhnqfm1si4rUvy0rFLdkQRD2Cphn9J_cYStNXYkBEfzp3qvr9KOiF8Dx-IMjoJ66OSehJFLz1ATFr1x26OqFWa9Emb7Ct0RoxxIiTOKgYwMBngGciKy7C8wsV5t3dIjTuiBD3fXJxFWusZYN7gHYldwm8A30ZGEveJmfI-loE-MjfSFw&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9tZWRpYS1jZG4uYXRsYXNzaWFuLmNvbS91cy13ZXN0LTIvdjIvY2RuL2NsaWVudC81YWRkYjc1NS01OWFjLTRhODUtOWY3Ny0zZTQwMDNkN2ViNmEvZmlsZS83ZjVkNTcyMi0xODQyLTQzNzEtYTZiMS1kYTFmM2M1NzFlNmUvYmluYXJ5PyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjQ2NDcyNjd9fX1dfQ__&Key-Pair-Id=K3CEQBLVYJW0FT&Signature=ENQRK5wteGDiy8PNLF6I4FpJqDP7CoBgG3dXUm-II4BMN4busovSqgQAx2kIpCDv0GySAFbfNpF4jiYjyNd-eUq6--IyuTsH1-kxg6gUDJLt-qOLhQp9OKTORJcUJgVGPmF3-GqtFFtWrPUxRcqT9Gp-mSGbi2iMG27xTGF4aHe8Dq4NcD~K6CvADZ--iSatVPSfnncsnxxBQkcrvgGUg3V2q4dCMIZ8zirBDO0P3ddbsGQ0hkUOqPZPDsiaWi6v3wK7sWPObKUADRkZCuBgHrZUbE-KP6mWkC7-iQugP2gTT9WJYE4gFwyoKvyz2nec2Z9k6fRiG6ThB-I~0tIv7Q__")!
    
    var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    
    var zipDestination: URL {
        documentsDirectory.appendingPathComponent("UIQAModel-Light.mlpackage.zip")
    }
    
    var modelDirectory: URL {
        documentsDirectory.appendingPathComponent("UIQAModel.mlpackage")
    }
    
    var body: some View {
        VStack(spacing: 20) {
            Text("CoreML Runtime Compilation")
                .font(.title)
                .bold()
            
            // Status
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: isProcessing ? "arrow.triangle.2.circlepath" : "checkmark.circle")
                        .foregroundColor(isProcessing ? .blue : .green)
                        .imageScale(.large)
                    
                    Text(statusMessage)
                        .font(.headline)
                }
                
                // Download progress bar
                if isDownloading {
                    VStack(alignment: .leading, spacing: 5) {
                        HStack {
                            Text("Downloading...")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(Int(downloadProgress * 100))%")
                                .font(.subheadline)
                                .foregroundColor(.blue)
                                .monospacedDigit()
                        }
                        
                        ProgressView(value: downloadProgress, total: 1.0)
                            .progressViewStyle(.linear)
                            .tint(.blue)
                    }
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(10)
                }
                
                if let score = predictionScore, let time = inferenceTime {
                    VStack(alignment: .leading, spacing: 5) {
                        if let image = selectedImage {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 200)
                                .cornerRadius(10)
                        }
                        
                        Text("Quality Score: \(String(format: "%.4f", score))")
                            .font(.title2)
                            .foregroundColor(.green)
                        Text("Inference Time: \(String(format: "%.4f", time))s")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(10)
                }
            }
            .padding()
            
            // Buttons
            VStack(spacing: 15) {
                Button(action: startProcess) {
                    HStack {
                        Image(systemName: "play.fill")
                        Text("Download, Compile & Predict")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(isProcessing ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(isProcessing)
                
                // Photo picker button (only show if model is compiled)
                if let _ = compiledModelURL {
                    PhotosPicker(selection: $selectedPhotoItem, matching: .images) {
                        HStack {
                            Image(systemName: "photo.on.rectangle.angled")
                            Text("Pick Photo from Library")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.purple)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .disabled(isProcessing)
                    .onChange(of: selectedPhotoItem) { newItem in
                        Task {
                            await loadAndPredictPhoto(newItem)
                        }
                    }
                }
                
                Button(action: clearCache) {
                    HStack {
                        Image(systemName: "trash")
                        Text("Clear Model Cache")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red.opacity(0.8))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(isProcessing)
            }
            .padding(.horizontal)
            
            // Logs
            ScrollView {
                VStack(alignment: .leading, spacing: 5) {
                    ForEach(Array(logs.enumerated()), id: \.offset) { _, log in
                        Text(log)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(logColor(for: log))
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
            }
            .background(Color.black.opacity(0.05))
            .cornerRadius(10)
            .padding(.horizontal)
        }
        .padding()
    }
    
    func logColor(for message: String) -> Color {
        if message.contains("✅") {
            return .green
        } else if message.contains("❌") {
            return .red
        } else if message.contains("⏳") || message.contains("📥") || message.contains("🔨") {
            return .blue
        } else {
            return .primary
        }
    }
    
    func addLog(_ message: String) {
        DispatchQueue.main.async {
            logs.append(message)
            print(message)
        }
    }
    
    func updateStatus(_ message: String) {
        DispatchQueue.main.async {
            statusMessage = message
        }
    }
    
    func clearCache() {
        let fileManager = FileManager.default
        do {
            if fileManager.fileExists(atPath: modelDirectory.path) {
                try fileManager.removeItem(at: modelDirectory)
                addLog("✅ Model cache cleared")
            }
            if fileManager.fileExists(atPath: zipDestination.path) {
                try fileManager.removeItem(at: zipDestination)
                addLog("✅ Zip file removed")
            }
            predictionScore = nil
            inferenceTime = nil
            downloadProgress = 0.0
            isDownloading = false
            compiledModelURL = nil
            predictor = nil
            selectedImage = nil
            selectedPhotoItem = nil
            updateStatus("Cache cleared")
        } catch {
            addLog("❌ Failed to clear cache: \(error.localizedDescription)")
        }
    }
    
    func startProcess() {
        isProcessing = true
        logs.removeAll()
        predictionScore = nil
        inferenceTime = nil
        downloadProgress = 0.0
        isDownloading = false
        updateStatus("Starting process...")
        
        Task {
            await performFullWorkflow()
        }
    }
    
    func performFullWorkflow() async {
        let fileManager = FileManager.default
        
        addLog("🌐 Model download URL: \(modelDownloadURL.absoluteString)")
        addLog("📁 Zip destination: \(zipDestination.path)")
        addLog("📁 Model destination: \(modelDirectory.path)")
        
        // Check if model already exists
        if fileManager.fileExists(atPath: modelDirectory.path) {
            addLog("✓ Model already exists locally, skipping download")
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: modelDirectory.path, isDirectory: &isDirectory) {
                if isDirectory.boolValue {
                    addLog("✓ Model package is valid")
                    await compileAndRun(modelPath: modelDirectory)
                    return
                }
            }
        }
        
        // Download model
        updateStatus("Downloading model...")
        addLog("📥 Starting download...")
        
        // Set downloading state
        DispatchQueue.main.async {
            self.isDownloading = true
            self.downloadProgress = 0.0
        }
        
        do {
            // Download with progress tracking
            let (localURL, response) = try await downloadWithProgress(from: modelDownloadURL)
            
            DispatchQueue.main.async {
                self.isDownloading = false
                self.downloadProgress = 1.0
            }
            
            addLog("✅ Download completed")
            
            guard let httpResponse = response as? HTTPURLResponse else {
                addLog("❌ Invalid response")
                updateStatus("Download failed")
                isProcessing = false
                return
            }
            
            if httpResponse.statusCode != 200 {
                addLog("❌ Server returned status code: \(httpResponse.statusCode)")
                updateStatus("Download failed")
                isProcessing = false
                return
            }
            
            // Move zip file
            if fileManager.fileExists(atPath: zipDestination.path) {
                try fileManager.removeItem(at: zipDestination)
            }
            try fileManager.moveItem(at: localURL, to: zipDestination)
            addLog("✅ Zip file saved")
            
            // Extract using SSZipArchive
            updateStatus("Extracting model...")
            addLog("📦 Extracting zip file...")
            
            let success = SSZipArchive.unzipFile(
                atPath: zipDestination.path,
                toDestination: documentsDirectory.path
            )
            
            if success {
                addLog("✅ Extraction completed!")
                
                // Verify extraction
                if fileManager.fileExists(atPath: modelDirectory.path) {
                    addLog("✅ Model package extracted successfully!")
                    
                    // Clean up zip
                    try? fileManager.removeItem(at: zipDestination)
                    addLog("🗑 Zip file removed")
                    
                    // Compile and run
                    await compileAndRun(modelPath: modelDirectory)
                } else {
                    addLog("❌ Model package not found after extraction")
                    updateStatus("Extraction verification failed")
                    isProcessing = false
                }
            } else {
                addLog("❌ Extraction failed")
                updateStatus("Extraction failed")
                isProcessing = false
            }
            
        } catch {
            addLog("❌ Download/Extraction error: \(error.localizedDescription)")
            updateStatus("Failed")
            isProcessing = false
            isDownloading = false
        }
    }
    
    // Download with progress tracking
    func downloadWithProgress(from url: URL) async throws -> (URL, URLResponse) {
        let (asyncBytes, response) = try await URLSession.shared.bytes(from: url)
        
        // Get expected content length
        let expectedLength = response.expectedContentLength
        
        // Create temporary file
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        
        // Download with progress
        var receivedData = Data()
        var receivedLength: Int64 = 0
        
        for try await byte in asyncBytes {
            receivedData.append(byte)
            receivedLength += 1
            
            // Update progress every 100KB to avoid too frequent UI updates
            if receivedLength % 100_000 == 0 && expectedLength > 0 {
                let progress = Double(receivedLength) / Double(expectedLength)
                await MainActor.run {
                    self.downloadProgress = progress
                }
            }
        }
        
        // Final progress update
        if expectedLength > 0 {
            let progress = Double(receivedLength) / Double(expectedLength)
            await MainActor.run {
                self.downloadProgress = progress
            }
        }
        
        // Write to temp file
        try receivedData.write(to: tempURL)
        
        return (tempURL, response)
    }
    
    // Load and predict photo from PhotosPicker
    func loadAndPredictPhoto(_ item: PhotosPickerItem?) async {
        guard let item = item else { return }
        
        isProcessing = true
        addLog("\n📸 Loading selected photo...")
        updateStatus("Loading photo...")
        
        do {
            guard let imageData = try await item.loadTransferable(type: Data.self) else {
                addLog("❌ Failed to load image data")
                updateStatus("Failed to load image")
                isProcessing = false
                return
            }
            
            guard let uiImage = UIImage(data: imageData) else {
                addLog("❌ Failed to create UIImage from data")
                updateStatus("Invalid image format")
                isProcessing = false
                return
            }
            
            addLog("✅ Photo loaded successfully")
            addLog("   Size: \(Int(uiImage.size.width))x\(Int(uiImage.size.height))")
            
            // Check predictor on main actor
            let hasPredictor = await MainActor.run { self.predictor != nil }
            
            // Run prediction with the selected photo
            if hasPredictor {
                await runPrediction(customImage: uiImage)
            } else {
                addLog("❌ Predictor not initialized yet")
                updateStatus("Model not ready")
                isProcessing = false
            }
            
        } catch {
            addLog("❌ Error loading photo: \(error.localizedDescription)")
            updateStatus("Failed to load photo")
            isProcessing = false
        }
    }
    
    func compileAndRun(modelPath: URL) async {
        updateStatus("Compiling model...")
        addLog("\n🔨 Compiling model at: \(modelPath.path)")
        
        do {
            let compiledURL = try await MLModel.compileModel(at: modelPath)
            addLog("✅ Model compiled successfully!")
            addLog("📦 Compiled model at: \(compiledURL.path)")
            
            // Create predictor once with compiled model
            let newPredictor = UIQACoreMLPredictor(modelPath: compiledURL)
            
            // Store compiled model URL and predictor for later use
            await MainActor.run {
                self.compiledModelURL = compiledURL
                self.predictor = newPredictor
            }
            
            addLog("✅ Predictor initialized")
            
            await runPrediction(useTestImage: true)
            
        } catch {
            addLog("❌ Model compilation failed: \(error.localizedDescription)")
            updateStatus("Compilation failed")
            isProcessing = false
        }
    }
    
    func runPrediction(useTestImage: Bool = false, customImage: UIImage? = nil) async {
        // Extract predictor on main actor first
        let currentPredictor = await MainActor.run { self.predictor }
        
        guard let predictor = currentPredictor else {
            addLog("❌ Predictor not initialized")
            updateStatus("Predictor not ready")
            isProcessing = false
            return
        }
        
        updateStatus("Running prediction...")
        addLog("\n🔮 Running prediction...")
        
        var imageToPredict: UIImage?
        
        // Determine which image to use
        if let custom = customImage {
            imageToPredict = custom
            addLog("✓ Using custom selected image")
        } else if useTestImage {
            // Load image from asset catalog
            guard let testImage = UIImage(named: "forest") else {
                addLog("❌ Could not load test image 'forest' from asset catalog")
                addLog("   Make sure 'forest' image is in Media.xcassets")
                updateStatus("Image loading failed")
                isProcessing = false
                return
            }
            imageToPredict = testImage
            addLog("✓ Using test image from asset catalog")
        } else {
            addLog("❌ No image provided for prediction")
            updateStatus("No image selected")
            isProcessing = false
            return
        }
        
        guard let image = imageToPredict else {
            addLog("❌ Failed to get image for prediction")
            updateStatus("Image error")
            isProcessing = false
            return
        }
        
        // Store the image being predicted
        await MainActor.run {
            self.selectedImage = image
        }
        
        do {
            let (score, time) = try await predictor.predict(image)
            
            addLog("🎉 Prediction successful!")
            addLog("   Score: \(score)")
            addLog("   Inference Time: \(String(format: "%.4f", time))s")
            
            DispatchQueue.main.async {
                self.predictionScore = score
                self.inferenceTime = time
                self.updateStatus("Completed successfully!")
                self.isProcessing = false
            }
            
        } catch {
            addLog("❌ Prediction error: \(error.localizedDescription)")
            updateStatus("Prediction failed")
            isProcessing = false
        }
    }
}
