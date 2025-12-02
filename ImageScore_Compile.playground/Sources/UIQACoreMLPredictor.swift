import UIKit
import CoreML
import Vision
#if canImport(MLCompute)
import MLCompute
#endif

/// Actor to manage thread-safe model initialization and access
@available(iOS 16.0, *)
private actor ModelManager {
    private var model: UIQAModel?
    private var isInitialized: Bool = false
    private var modelPath: URL?
    
    /// Sets the model path for loading
    func setModelPath(_ path: URL) {
        self.modelPath = path
    }
    
    /// Gets the model, initializing it if necessary (only once per session)
    func getModel() throws -> UIQAModel {
        // Return cached model if already initialized
        if isInitialized, let existingModel = model {
            #if DEBUG
            print("[BestPhoto] \(Date()) == ♻️ Returning cached model from actor")
            #endif
            return existingModel
        }
        
        // Initialize model only if not already done
        if !isInitialized {
            #if DEBUG
            print("[BestPhoto] \(Date()) == 🔒 Actor initializing model for the first time...")
            #endif
            
            let newModel = try initializeModel()
            model = newModel
            isInitialized = true
            
            #if DEBUG
            print("[BestPhoto] \(Date()) == ✅ Model successfully initialized and cached in actor")
            #endif
            
            return newModel
        }
        
        // Fallback (should not reach here)
        guard let existingModel = model else {
            throw UIQACoreMLPredictor.UIQAModelError.unknownError
        }
        
        return existingModel
    }
    
    /// Checks if the device has Neural Engine support at runtime
    /// - Returns: True if Neural Engine is available
    private func hasNeuralEngine() -> Bool {
        #if canImport(MLCompute)
        if let _ = MLCDevice.ane() {
            return true
        }
        #endif
        return false
    }
    
    /// Checks if the code is running in a playground environment
    /// - Returns: True if running in a playground
    private func isRunningInPlayground() -> Bool {
        let bundlePath = Bundle.main.bundlePath
        let processName = ProcessInfo.processInfo.processName
        
        // Check if bundle path or process name contains "Playground"
        return bundlePath.contains(".playground") || 
               processName.contains("Playground") ||
               bundlePath.contains("XCPGDevices")
    }
    
    /// Determines the optimal compute units based on device hardware capabilities and iOS version
    /// - Returns: The optimal MLComputeUnits configuration
    private func getOptimalComputeUnits() -> MLComputeUnits {
        // If running in playground, use CPU only for better compatibility
        if isRunningInPlayground() {
            #if DEBUG
            print("[BestPhoto] \(Date()) == 🎮 Running in Playground - Using CPU only for compatibility")
            #endif
            return .cpuOnly
        }
        
        let hasANE = hasNeuralEngine()
        let iosVersion = ProcessInfo.processInfo.operatingSystemVersion
        
        #if DEBUG
        let deviceModel = getDeviceModel()
        print("[BestPhoto] \(Date()) == 🔍 Device Model: \(deviceModel)")
        print("[BestPhoto] \(Date()) == 🔍 iOS: \(iosVersion.majorVersion).\(iosVersion.minorVersion).\(iosVersion.patchVersion)")
        print("[BestPhoto] \(Date()) == 🔍 Neural Engine: \(hasANE ? "✅ Available" : "❌ Not Available")")
        #endif
        
        // Use Neural Engine only on iOS 18+ with ANE support, otherwise use GPU
        if iosVersion.majorVersion >= 18 && hasANE {
            #if DEBUG
            print("[BestPhoto] \(Date()) == 🧠 Using CPU + Neural Engine (iOS 18+ with ANE)")
            #endif
            return .cpuAndNeuralEngine
        } else {
            #if DEBUG
            if hasANE {
                print("[BestPhoto] \(Date()) == 🎮 Using CPU + GPU (iOS < 18, ANE not used)")
            } else {
                print("[BestPhoto] \(Date()) == 🎮 Using CPU + GPU (no ANE hardware)")
            }
            #endif
            return .cpuAndGPU
        }
    }
    
    /// Gets the device model identifier (e.g., "iPhone16,1")
    /// - Returns: Device model identifier string
    private func getDeviceModel() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machineMirror = Mirror(reflecting: systemInfo.machine)
        let identifier = machineMirror.children.reduce("") { identifier, element in
            guard let value = element.value as? Int8, value != 0 else { return identifier }
            return identifier + String(UnicodeScalar(UInt8(value)))
        }
        return identifier
    }
    
    /// Initializes the Core ML model with proper error handling and timeout fallback
    /// If loading takes more than 2 minutes, retries with CPU only
    private func initializeModel() throws -> UIQAModel {
        #if DEBUG
        print("[BestPhoto] \(Date()) == 🔄 Starting UIQA Core ML model initialization...")
        #endif
        
        // Check if model file exists at the specified path
        if let modelPath = modelPath {
            #if DEBUG
            print("Looking for model at: \(modelPath.path)")
            #endif
            
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: modelPath.path) {
                #if DEBUG
                print("✓ Model file found at: \(modelPath.path)")
                #endif
            } else {
                #if DEBUG
                print("❌ Model file not found at: \(modelPath.path)")
                #endif
                throw UIQACoreMLPredictor.UIQAModelError.configurationError("Model file not found at path: \(modelPath.path)")
            }
        } else {
            #if DEBUG
            print("❌ Model path not set!")
            #endif
            throw UIQACoreMLPredictor.UIQAModelError.configurationError("Model path not set")
        }
        
        // First attempt with optimal compute units
        let config = MLModelConfiguration()
        config.computeUnits = getOptimalComputeUnits()
        config.allowLowPrecisionAccumulationOnGPU = true
        
        #if DEBUG
        print("[BestPhoto] \(Date()) == ⚙️ Model configuration set: computeUnits = \(config.computeUnits.rawValue)")
        print("[BestPhoto] \(Date()) == 💾 Available memory: \(ProcessInfo.processInfo.physicalMemory / 1024 / 1024) MB")
        #endif
        
        let startTime = Date()
        let timeout: TimeInterval = 120.0 // 2 minutes
        
        // Try loading with optimal configuration
        let model = try loadModelWithTimeout(configuration: config, timeout: timeout, startTime: startTime)
        
        #if DEBUG
        let loadTime = Date().timeIntervalSince(startTime)
        print("[BestPhoto] \(Date()) == ✅ Successfully loaded UIQA Core ML model (took \(String(format: "%.2f", loadTime))s)")
        #endif
        
        return model
    }
    
    /// Attempts to load the model with a timeout and fallback to CPU only
    /// - Parameters:
    ///   - configuration: The MLModelConfiguration to use
    ///   - timeout: Timeout in seconds (default 120 seconds = 2 minutes)
    ///   - startTime: Start time for tracking total load time
    /// - Returns: Loaded UIQAModel
    /// - Throws: UIQAModelError if loading fails
    private func loadModelWithTimeout(configuration: MLModelConfiguration, timeout: TimeInterval, startTime: Date) throws -> UIQAModel {
        // Extract values from configuration before async closure to avoid Sendable issues
        let computeUnits = configuration.computeUnits
        let allowLowPrecision = configuration.allowLowPrecisionAccumulationOnGPU
        
        #if DEBUG
        print("[BestPhoto] \(Date()) == 🔄 Attempting to load model with computeUnits: \(computeUnits.rawValue)")
        #endif
        
        var loadResult: Result<UIQAModel, Error>?
        let semaphore = DispatchSemaphore(value: 0)
        
        // Load model using Task on global queue
        Task {
            do {
                // Create configuration inside the async block
                let config = MLModelConfiguration()
                config.computeUnits = computeUnits
                config.allowLowPrecisionAccumulationOnGPU = allowLowPrecision
                
                // Load model from the specified path
                let model: UIQAModel
                if let modelPath = self.modelPath {
                    print("Loading model from path:\(modelPath)")
                    model = try UIQAModel(contentsOf: modelPath, configuration: config)
                } else {
                    // Fallback to bundle loading
                    model = try UIQAModel(configuration: config)
                }
                loadResult = .success(model)
                
                #if DEBUG
                print("[BestPhoto] \(Date()) == 🎯 Model loaded successfully")
                #endif
            } catch {
                loadResult = .failure(error)
                #if DEBUG
                print("[BestPhoto] \(Date()) == ❌ Model loading failed: \(error.localizedDescription)")
                #endif
            }
            semaphore.signal()
        }
        
        // Wait with timeout
        let result = semaphore.wait(timeout: .now() + timeout)
        
        if result == .timedOut {
            #if DEBUG
            let elapsedTime = Date().timeIntervalSince(startTime)
            print("[BestPhoto] \(Date()) == ⏱ Model loading timed out after \(String(format: "%.2f", elapsedTime))s")
            print("[BestPhoto] \(Date()) == 🔄 Retrying with CPU only as fallback...")
            #endif
            
            // Fallback to CPU only
            return try loadModelWithCPUOnly(startTime: startTime)
        }
        
        // Check result
        guard let result = loadResult else {
            throw UIQACoreMLPredictor.UIQAModelError.unknownError
        }
        
        switch result {
        case .success(let model):
            return model
        case .failure(let error):
            #if DEBUG
            print("[BestPhoto] \(Date()) == ❌ Error occurred during loading, attempting CPU fallback... \(error.localizedDescription)")
            #endif
            
            // Fallback to CPU only on error
            return try loadModelWithCPUOnly(startTime: startTime)
        }
    }
    
    /// Loads the model with CPU only configuration as a fallback
    /// - Parameter startTime: Start time for tracking total load time
    /// - Returns: Loaded UIQAModel
    /// - Throws: UIQAModelError if loading fails
    private func loadModelWithCPUOnly(startTime: Date) throws -> UIQAModel {
        #if DEBUG
        print("[BestPhoto] \(Date()) == 💻 Loading model with CPU only...")
        #endif
        
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly
        
        do {
            // Load model from the specified path or fallback to bundle
            let model: UIQAModel
            if let modelPath = self.modelPath {
                model = try UIQAModel(contentsOf: modelPath, configuration: cpuConfig)
            } else {
                model = try UIQAModel(configuration: cpuConfig)
            }
            
            #if DEBUG
            let totalTime = Date().timeIntervalSince(startTime)
            print("[BestPhoto] \(Date()) == ✅ Model loaded successfully with CPU only (total time: \(String(format: "%.2f", totalTime))s)")
            #endif
            
            return model
        } catch {
            #if DEBUG
            print("[BestPhoto] \(Date()) == ❌ Failed to load model even with CPU only: \(error.localizedDescription)")
            #endif
            throw UIQACoreMLPredictor.UIQAModelError.modelInitializationFailed(error)
        }
    }
}

/// A predictor that uses the Core ML version of the UIQA Model
@available(iOS 16.0, *)  // Matches minimum_deployment_target=ct.target.iOS16
public class UIQACoreMLPredictor: MDBestPhotoPredictor {
    private var isRunning: Bool = false
    
    /// Public initializer
    public init() {}
    
    /// Public initializer with model path (for playground use)
    public init(modelPath: URL) {
        Task {
            await modelManager.setModelPath(modelPath)
        }
    }
    
    /// Error types specific to UIQA model loading
    enum UIQAModelError: Error {
        case modelInitializationFailed(Error)
        case configurationError(String)
        case unknownError
        
        var description: String {
            switch self {
            case .modelInitializationFailed(let error):
                return "Failed to initialize UIQA model: \(error.localizedDescription)"
            case .configurationError(let message):
                return "Configuration error: \(message)"
            case .unknownError:
                return "Unknown error occurred while loading UIQA model"
            }
        }
    }
    
    /// Actor managing thread-safe model access
    private let modelManager = ModelManager()
    
    /// Gets the model using the actor-managed thread-safe access
    /// This method bridges synchronous and asynchronous contexts
    private func getModel() throws -> UIQAModel {
        // Use a semaphore to bridge async actor call to sync method
        var result: Result<UIQAModel, Error>?
        let semaphore = DispatchSemaphore(value: 0)
        
        Task {
            do {
                let model = try await modelManager.getModel()
                result = .success(model)
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }
        
        semaphore.wait()
        
        switch result {
        case .success(let model):
            return model
        case .failure(let error):
            throw error
        case .none:
            throw UIQAModelError.unknownError
        }
    }
    
    /// Error types specific
    enum PredictionError: Error {
        case preprocessingFailed
        case predictionFailed
        case alreadyRunning
        
        var localizedDescription: String {
            switch self {
            case .preprocessingFailed:
                return "Failed to preprocess the input image"
            case .predictionFailed:
                return "Failed to run model prediction"
            case .alreadyRunning:
                return "Prediction is already in progress"
            }
        }
    }
    
    /// Constants for the UIQA Model
    private enum Constants {
        static let cropSize = 480    // Size for all branches
        static let nFragment = 15    // Number of fragments for distortion branch
        static let fSize = 32        // Fragment size for distortion branch
        
        // Normalization constants
        static let mean: [Float] = [0.485, 0.456, 0.406]
        static let std: [Float] = [0.229, 0.224, 0.225]
    }
    
    /// Predicts image quality score for the given image
    /// - Parameter image: Input UIImage to assess
    /// - Returns: Tuple containing quality score and inference time in seconds
    /// - Throws: PredictionError if any step fails
    public func predict(_ image: UIImage) throws -> (score: Float, inferenceTime: Double) {
        // Check if prediction is already running
        if isRunning {
            throw PredictionError.alreadyRunning
        }
        
        isRunning = true
        let startTime = CACurrentMediaTime()
        
        defer {
            isRunning = false
        }
        
        guard let model = try? getModel() else {
            throw PredictionError.predictionFailed
        }
        
        // Center crop the image to square
        let shortSide = min(image.size.width, image.size.height)
        let cropRect = CGRect(
            x: (image.size.width - shortSide) / 2,
            y: (image.size.height - shortSide) / 2,
            width: shortSide,
            height: shortSide
        )
        
        // Prepare inputs for all three branches
//        guard let croppedImage = image.sd_resizedImage(with: cropRect.size, scaleMode: .aspectFill) else {
//            throw PredictionError.preprocessingFailed
//        }
        
        // 2 - Resize image to 480x480 (required input size for UIQA)
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 480, height: 480), true, 2.0)
        image.draw(in: CGRect(x: 0, y: 0, width: 480, height: 480))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        //let resizedImage = croppedImage//croppedImage.resize(to: CGSize(width: Constants.cropSize, height: Constants.cropSize))
        
        // Convert to pixel buffers
        guard let resizedImage,
              let aestheticsBuffer = resizedImage.pixelBuffer(),
              let distortionBuffer = resizedImage.pixelBuffer(),
              let saliencyBuffer = resizedImage.pixelBuffer()
        else {
            throw PredictionError.preprocessingFailed
        }
        
        // Run prediction with pixel buffers directly
        // Note: The model now handles normalization internally
        do {
            let output = try model.prediction(
                input_aesthetics: aestheticsBuffer,
                input_distortion: distortionBuffer,
                input_saliency: saliencyBuffer
            )
            
            let score = output.quality_score[0].floatValue
            let inferenceTime = CACurrentMediaTime() - startTime
            return (score: score, inferenceTime: inferenceTime)
        } catch let e {
            print("[BestPhoto] predict failed:\(e.localizedDescription)")
            throw PredictionError.predictionFailed
        }
    }
    
    /// Predicts image quality score with a completion handler
    /// - Parameters:
    ///   - image: Input UIImage to assess
    ///   - completion: Completion handler with Result type
    func predictAsync(_ image: UIImage, completion: @escaping (Result<(score: Float, inferenceTime: Double), Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let result = try self.predict(image)
                DispatchQueue.main.async {
                    completion(.success(result))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
    
    /// Enum representing different branches of the UIQA model
    enum Branch {
        case aesthetics
        case distortion
        case saliency
    }
    
    public func tryLoadCoreMLModel() {
        #if DEBUG
        print("[BestPhoto] \(Date()) == 🚀 Starting Core ML model loading process")
        print("[BestPhoto] \(Date()) == 📱 Device Info:")
        print("[BestPhoto] \(Date()) == 📱 iOS Version: \(UIDevice.current.systemVersion)")
        print("[BestPhoto] \(Date()) == 📱 Device Model: \(UIDevice.current.model)")
        print("[BestPhoto] \(Date()) == 📱 Process Name: \(ProcessInfo.processInfo.processName)")
        print("[BestPhoto] \(Date()) == 📱 Process ID: \(ProcessInfo.processInfo.processIdentifier)")
        
        // Log Core ML environment info
        print("[BestPhoto] \(Date()) == 📱 Core ML Info:")
        print("[BestPhoto] \(Date()) == 📱 Model minimum deployment target: iOS 16.0")
        print("[BestPhoto] \(Date()) == 📱 Current device iOS: \(UIDevice.current.systemVersion)")
        
        // Log Core ML configuration
        let config = MLModelConfiguration()
        print("[BestPhoto] \(Date()) == 📱 Core ML compute units: \(config.computeUnits.rawValue)")
        print("[BestPhoto] \(Date()) == 📱 Core ML allows low precision: \(config.allowLowPrecisionAccumulationOnGPU)")
        if let metalDevice = MTLCreateSystemDefaultDevice() {
            print("[BestPhoto] \(Date()) == 📱 Metal device name: \(metalDevice.name)")
        }
        
        // Check if we're running on iOS 16 or later
        if ProcessInfo().isOperatingSystemAtLeast(OperatingSystemVersion(majorVersion: 16, minorVersion: 0, patchVersion: 0)) {
            print("[BestPhoto] \(Date()) == ✅ Device meets minimum iOS requirement (iOS 16.0+)")
        } else {
            print("[BestPhoto] \(Date()) == ⚠️ Device iOS version (\(UIDevice.current.systemVersion)) is below minimum requirement (iOS 16.0)")
        }
        #endif
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                #if DEBUG
                print("[BestPhoto] \(Date()) == ⚠️ Self is nil while trying to load UIQA Core ML model")
                #endif
                return
            }
            
            #if DEBUG
            print("[BestPhoto] \(Date()) == 🧵 Loading on thread: \(Thread.current.description)")
            #endif
            
            do {
                let startTime = Date()
                
                #if DEBUG
                print("[BestPhoto] \(Date()) == 🔄 Initiating model load sequence...")
                #endif
                
                let _ = try self.getModel() // This will trigger model initialization if needed
                
                #if DEBUG
                let loadTime = Date().timeIntervalSince(startTime)
                print("[BestPhoto] \(Date()) == 🎉 UIQA Core ML model loaded successfully!")
                print("[BestPhoto] \(Date()) == ⏱ Total loading time: \(String(format: "%.2f", loadTime))s")
                
                // Verify compute environment
                let config = MLModelConfiguration()
                print("[BestPhoto] \(Date()) == 💻 Compute environment details:")
                print("[BestPhoto] \(Date()) == 💻 Available compute units: \(config.computeUnits.rawValue)")
                print("[BestPhoto] \(Date()) == 💻 Allows low precision: \(config.allowLowPrecisionAccumulationOnGPU)")
                print("[BestPhoto] \(Date()) == 💻 Preferred metaldevice: \(String(describing: MTLCreateSystemDefaultDevice()?.name))")
                #endif
                
            } catch UIQAModelError.modelInitializationFailed(let error) {
                #if DEBUG
                print("[BestPhoto] \(Date()) == ❌ Model initialization failed")
                print("[BestPhoto] \(Date()) == ❌ Error type: \(type(of: error))")
                print("[BestPhoto] \(Date()) == ❌ Error description: \(error.localizedDescription)")
                print("[BestPhoto] \(Date()) == 📋 Full error details: \(String(describing: error))")
                print("[BestPhoto] \(Date()) == 📋 Stack trace:")
                Thread.callStackSymbols.forEach { symbol in
                    print("[BestPhoto] \(Date()) == 📋 \(symbol)")
                }
                #endif
            } catch {
                #if DEBUG
                print("[BestPhoto] \(Date()) == ❌ Unexpected error while loading model")
                print("[BestPhoto] \(Date()) == ❌ Error type: \(type(of: error))")
                print("[BestPhoto] \(Date()) == ❌ Error description: \(error.localizedDescription)")
                print("[BestPhoto] \(Date()) == 📋 Full error details: \(String(describing: error))")
                print("[BestPhoto] \(Date()) == 📋 Stack trace:")
                Thread.callStackSymbols.forEach { symbol in
                    print("[BestPhoto] \(Date()) == 📋 \(symbol)")
                }
                #endif
            }
        }
    }
}
