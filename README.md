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

## For full tutorial, visit this link. 

https://hadiajalil.com/coreml-compilingmodel/
