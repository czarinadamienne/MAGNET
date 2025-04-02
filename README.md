# MAGNET
# Group 6 - Integrating Project
## YOUTUBE LINK: 

- This project aimts to implement the MAGNET DNA pre-alignment filter in C and utilize CUDA programming to parallelize certain functions in the source code. The CUDA implementation uses shared memory concept, memory management and atomic operations. This is tested through using 1000 genomic sequences with a length of 100 characters. The filter was tested with edit distance threshold of 0, 3, and 10.
  
## Members

* Alamay, Carl Justine
* Ang, Czarina Damienne
* Esteban, Janina Angela
* Herrera, Diego Martin

## Report
### I. Screenshot of the program output with execution time and correctness check (C)
   **E = 0**
   ![image](https://github.com/user-attachments/assets/d0c287cf-4a82-4dd2-b055-f04cf57b7dfe)

   **E = 3**
   ![image](https://github.com/user-attachments/assets/7521241d-e034-4f61-9406-3381d97bd2eb)

   **E = 8**
   ![image](https://github.com/user-attachments/assets/7720c52c-fe0e-4927-bed4-d5e7188a015c)

   
<br/>
### II. Screenshot of the program output with execution time and correctness check (CUDA)
  **E = 0**<br/>
    **256 Blocks**<br/>
    
  ![image](https://github.com/user-attachments/assets/3c21103a-259d-443a-94a3-8307fe1a75da)

  **E = 3**<br/>
    **256 Blocks**
    ![image](https://github.com/user-attachments/assets/dd030d92-e0e4-4f6b-b7a6-c75fbfc68249)

  **E = 8**<br/>
    **256 Blocks**
    ![image](https://github.com/user-attachments/assets/3b47e7e3-9847-4c19-b5d8-d3feac83b1bc)

<br/>
**E = 0**<br/>
    **1024 Blocks**<br/>
    
  ![image](https://github.com/user-attachments/assets/97860dca-c34e-4b92-a0ab-31bb5ebd662a)
    
**E = 3**<br/>
    **1024 Blocks**
    ![image](https://github.com/user-attachments/assets/2aeb2a45-f111-4c25-acca-d11738f84998)

**E = 8**<br/>
    **1024 Blocks**
    ![image](https://github.com/user-attachments/assets/27980cfb-5389-4e40-94f8-ae6dab330132)

  
<br/>
### III. Screenshot of the program implementation
**1. C Implementation**
   - Consecutive Zeros Info function<br/>

   - Extraction Encapsulation function<br/>
   
   - MAGNET function<br/>
     
   - Correctness checker<br/>
     

  text
**2. CUDA Implementation**
   - Consecutive Zeros Info function<br/>

   - Extraction Encapsulation function<br/>
   
   - MAGNET function<br/>
     
   - Correctness checker<br/>
     

     text
     
### IV. Comparative table of execution time and Analysis
Average execution time of C Program
| Dataset size | E = 0      | E = 3 | E = 10 |
| ------------ | ---------- | ----- | ------ |
| 1000         |  ms | ms | ms |


Average execution times of CUDA
| Dataset size = 1000 | E = 0 | E = 3| E = 10 | Speedup compared to C |
| ------------------- | ----- |------|--------| --------------------- |
| Threads = 256         |  ms |    ms |       ms  |      x  |
| Threads = 1024        |  ms |    ms |        ms |       x  |


Both kernels were timed with a vector size of 2^28 and their average execution times after 30 loops of the function were compared. The C kernel had an average execution time of 1667.850767 ms while the CUDA kernel with 1024 threads had an average execution time of 98.698222 ms. The CUDA kernel with 1024 threads was around 16.9 times faster than the C kernel. Additionally, the CUDA kernel with 256 threads had an execution time of 96.590928 ms, which is a 17.3 times faster than the C kernel. It is also observed that 256 threads is faster, for this case, than 1024 threads. This could be because 256 threads is the enough amount of threads needed as there is less memory access and fewer conflicts, since shared memory is applied in the code. There could be bank conflicts if there are too many threads per block.
<br/>

The C kernel is generally slower in execution due to the sequential execution of the function during the loop. The CPU will have to iterate 2^28 elements one at a time. The CUDA kernel deploys multiple threads for the computation of the histogram bins to allow for parallel execution of the function. The use of unified memory, page creation, and mem advise also aided in the reduction of execution time for the CUDA kernel. The unified memory removes the manual copying of data between host and device, page creation to reduce the number of page faults that occur, and mem advise to only send the output of the GPU back to the CPU rather than both the input and the output. These three techniques lowers the overall overhead of the kernel and allows for faster execution time.
<br/>

The shared memory concept was also applied in the creation of the CUDA program. Instead of each thread block having to access the global memory, it instead accesses its own shared memory which stores frequently accessed data. Each thread will update its own copy in their own shared memory block which is later merged with the global memory later on. This removes the latency that comes with repeatedly accessing global memory which is not present when accessing shared memory. This further reduced the memory overhead that the CUDA kernel experiences.
<br/>

### V. Discussion
- **Why is CUDA faster?** <br/>
The C Kernel runs on the CPU, which processes each data point one-by-one. If you have around 268 million items, it will definitely take a long time.
CUDA Kernel runs on GPU which splits the work into many smaller tasks and does them all at the same time as explained earlier, hence why the GPU finishes task much faster. CUDA has smart tricks such as unified memory, page creation, and mem advise. 
  - Unified Memory: Instead of manually moving data between CPU and GPU, the system does it automatically.
  - Page Creation: This reduces the number of interruptions (page faults) when the GPU needs new data
  - Mem Advise: Only sends back the final result from GPU to CPU, instead of sending everything which causes unnecessary data transfers.

- **Problems encountered** <br/>
Race conditions occurred with the threads in a block when writing in the shared memory. This caused incorrect values to be added to each histogram bin which produced the incorrect results. Multiple threads writing to the same memory location causing incorrect incrementation of some bins resulting in incorrect results.

- **AHA moments** <br/>
The use of the __syncthreads() function was implemented to allow for all threads in a block to finish updating the shared memory before merging with the global memory. The additional use of atomicAdd() to prevent two or more threads from updating the same memory location, preventing incorrect results.
