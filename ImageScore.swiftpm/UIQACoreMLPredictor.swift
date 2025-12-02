import UIKit
import CoreML
import Vision
#if canImport(MLCompute)
import MLCompute
#endif

// Mark MLModel as Sendable since CoreML models are thread-safe for concurrent inference
extension MLModel: @unchecked Sendable {}

/// Actor to manage thread-safe model initialization and access
@available(iOS 16.0, *)
private actor ModelManager {
    private var model: MLModel?
    private var isInitialized: Bool = false
    private var modelPath: URL?
    
    /// Initializes with a model path
    init(modelPath: URL? = nil) {
        self.modelPath = modelPath
    }
    
    /// Sets the model path for loading
    func setModelPath(_ path: URL) {
        self.modelPath = path
    }
    
    /// Gets the model, initializing it if necessary (only once per session)
    func getModel() throws -> MLModel {
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
//#if DEBUG
//        print("[BestPhoto] \(Date()) == 🎮 Running in Playground - Using CPU only for compatibility")
//#endif
//        //test
//        return .cpuOnly
        
        // If running in playground, use CPU only for better compatibility
//        if isRunningInPlayground() {
//            #if DEBUG
//            print("[BestPhoto] \(Date()) == 🎮 Running in Playground - Using CPU only for compatibility")
//            #endif
//            return .cpuOnly
//        }
        
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
    /// NOTE: runtime loading core ML must in main thread!!!
    private func initializeModel() throws -> MLModel {
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
        //config.allowLowPrecisionAccumulationOnGPU = true //test
        
        #if DEBUG
        print("[BestPhoto] \(Date()) == ⚙️ Model configuration set: computeUnits = \(config.computeUnits.rawValue)")
        print("[BestPhoto] \(Date()) == 💾 Available memory: \(ProcessInfo.processInfo.physicalMemory / 1024 / 1024) MB")
        #endif
        
        let startTime = Date()
        let timeout: TimeInterval = 120.0 // 2 minutes
        
        // Try loading with optimal configuration
        let model = try loadModel(configuration: config, startTime: startTime)
        
        #if DEBUG
        let loadTime = Date().timeIntervalSince(startTime)
        print("[BestPhoto] \(Date()) == ✅ Successfully loaded UIQA Core ML model (took \(String(format: "%.2f", loadTime))s)")
        #endif
        
        return model
    }
    
    /// Attempts to load the model synchronously
    /// - Parameters:
    ///   - configuration: The MLModelConfiguration to use
    ///   - startTime: Start time for tracking total load time
    /// - Returns: Loaded MLModel
    /// - Throws: UIQAModelError if loading fails
    private func loadModel(configuration: MLModelConfiguration, startTime: Date) throws -> MLModel {
        // Extract values from configuration
        let computeUnits = configuration.computeUnits
        
        #if DEBUG
        print("[BestPhoto] \(Date()) == 🔄 Attempting to load model with computeUnits: \(computeUnits.rawValue)")
        #endif
        
        do {
            // Create configuration
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits
            //config.allowLowPrecisionAccumulationOnGPU = allowLowPrecision
            
            // Load model from the specified path
            let model: MLModel
            if let modelPath = self.modelPath {
                print("Loading model from path:\(modelPath)")
                model = try MLModel(contentsOf: modelPath, configuration: config)
            } else {
                // Fallback to bundle loading (this will fail if model not in bundle)
                throw UIQACoreMLPredictor.UIQAModelError.configurationError("Model path not set")
            }
            
            #if DEBUG
            print("[BestPhoto] \(Date()) == 🎯 Model loaded successfully")
            #endif
            
            return model
        } catch {
            #if DEBUG
            print("[BestPhoto] \(Date()) == ❌ Model loading failed: \(error.localizedDescription)")
            #endif
            
            // Try CPU fallback
            #if DEBUG
            print("[BestPhoto] \(Date()) == ❌ Error occurred during loading, attempting CPU fallback... \(error.localizedDescription)")
            #endif
            
            return try loadModelWithCPUOnly(startTime: startTime)
        }
    }
    
    /// Loads the model with CPU only configuration as a fallback
    /// - Parameter startTime: Start time for tracking total load time
    /// - Returns: Loaded MLModel
    /// - Throws: UIQAModelError if loading fails
    private func loadModelWithCPUOnly(startTime: Date) throws -> MLModel {
        #if DEBUG
        print("[BestPhoto] \(Date()) == 💻 Loading model with CPU only...")
        #endif
        
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly
        
        do {
            // Load model from the specified path or fallback to bundle
            let model: MLModel
            if let modelPath = self.modelPath {
                model = try MLModel(contentsOf: modelPath, configuration: cpuConfig)
            } else {
                throw UIQACoreMLPredictor.UIQAModelError.configurationError("Model path not set")
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
public class UIQACoreMLPredictor: MDBestPhotoPredictor, @unchecked Sendable {
    private var isRunning: Bool = false
    
    /// Actor managing thread-safe model access
    private let modelManager: ModelManager
    
    /// Public initializer
    public init() {
        self.modelManager = ModelManager()
    }
    
    /// Public initializer with model path (for playground use)
    public init(modelPath: URL) {
        self.modelManager = ModelManager(modelPath: modelPath)
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
    
    /// Gets the model using the actor-managed thread-safe access
    /// This method bridges synchronous and asynchronous contexts
    private func getModel() async throws -> MLModel {
        return try await modelManager.getModel()
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
        static let cropSize = 360    // Size for all branches //test //480
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
    public func predict(_ image: UIImage) async throws -> (score: Float, inferenceTime: Double) {
        // Check if prediction is already running
        if isRunning {
            throw PredictionError.alreadyRunning
        }
        
        isRunning = true
        let startTime = CACurrentMediaTime()
        
        defer {
            isRunning = false
        }
        
        let model = try await getModel()
        
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
        
        // 2 - Resize image to 360x360 (required input size for UIQA)
        UIGraphicsBeginImageContextWithOptions(CGSize(width: Constants.cropSize, height: Constants.cropSize), true, 2.0)
        image.draw(in: CGRect(x: 0, y: 0, width: Constants.cropSize, height: Constants.cropSize))
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
            // Create input features dictionary for generic MLModel
            let inputFeatures: [String: Any] = [
                "input_aesthetics": aestheticsBuffer,
                "input_distortion": distortionBuffer,
                "input_saliency": saliencyBuffer
            ]
            
            // Create MLFeatureProvider from dictionary
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputFeatures)
            
            // Run prediction
            let output = try model.prediction(from: inputProvider)
            
            // Extract quality_score from output
            guard let qualityScoreMultiArray = output.featureValue(for: "quality_score")?.multiArrayValue else {
                throw PredictionError.predictionFailed
            }
            
            let score = Float(truncating: qualityScoreMultiArray[0])
            let inferenceTime = CACurrentMediaTime() - startTime
            return (score: score, inferenceTime: inferenceTime)
        } catch let e {
            print("[BestPhoto] predict failed:\(e.localizedDescription)")
            throw PredictionError.predictionFailed
        }
    }
    
    /// Enum representing different branches of the UIQA model
    enum Branch {
        case aesthetics
        case distortion
        case saliency
    }
}
