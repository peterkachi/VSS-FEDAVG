The user can successfully run the code by following these steps:
1. Download three datasets and unzip them.
2. Create a folder named `adaptive_synchronization` and place all the code from the project into this folder.
3. In the file `datasource.py`, modify the data download path in the function `def get_datasets()` to the path where the data is located (around line from 257 to 278), `mediator_length = world_size` (client number) / `mediator_number`.
4. The user can modify hyper-parameters in the file `worker_process_as.py`, including the choice of data-model, the `D_ALPHA` value related to non-IID extent, synchronization frequency, input image dimensions (`image_size`), and the number of labels (`n_labels = class number + 1`), etc.
5. Modify the client's IP address in lines 4 and 5 of the file `train.sh`.
6. After all modifications are complete, navigate to the `adaptive_synchronization` folder in the command line and enter `bash train.sh` to run the code.
7. The provided code is run for random-allocation method. For our VSS-FedAvg, please comment out the code on line 325 in the `as_manager.py` file (Lines 325 and 326 contain examples of the methods for allocating 24 clients and 12 clients randomly, respectively.).

Additionally, during the attempt to run the code, various errors may occur, such as missing necessary library files. For specific error messages and run results, please check the details in the `Logs` folder. And we strongly recommend running our codes under CUDA environments.