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
### II. Screenshot of the program output with execution time and correctness check (CUDA) <br/>

**E = 0** <br/>
  **256 Blocks** <br/>
    
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
    ![image](https://github.com/user-attachments/assets/8f735bb9-5c3d-49e0-9508-0b503eadbd89)

The ConsecutiveZeros() function scans and finds groups of consecutive zeros within in array, more specifically in the specified sequence. It creates three variables to store the starting and ending positions, and the lengths of the zero groups. When a "1" appears as it's going through the array, it marks the end of the zero sequence and saves the details before looking for the next zeros group. It will then return all the collected information from these zero groups.

   - Extraction Encapsulation function<br/>
    ![image](https://github.com/user-attachments/assets/6db3d59f-20a3-4b74-9ffb-f103b71ffe96)

  ![image](https://github.com/user-attachments/assets/7bc4072a-a5e6-47ed-a251-718ba0394ec4)
 
   - MAGNET function<br/>
    ![image](https://github.com/user-attachments/assets/326bf040-8377-4b99-888f-31c1da944611)

  ![image](https://github.com/user-attachments/assets/1e60b933-3afb-4a2c-99dc-09a8ea446b9d)

  ![image](https://github.com/user-attachments/assets/43fac034-12d8-47de-845f-fee05d5bbff8)
    
   - Correctness checker<br/>
  ![image](https://github.com/user-attachments/assets/954d3b87-2b06-4825-81f8-f0c1fb61939f)
   

  text
  
**2. CUDA Implementation**
   - Compute Hamming and Errors Kernel<br/>
  ![image](https://github.com/user-attachments/assets/bf950964-8479-45cb-8c84-6af45d1ac1be)
  ![image](https://github.com/user-attachments/assets/9369f273-1d30-437c-80d0-d3851fb5ddde)

   - Find Consecutive Zeroes Kernel<br/>
   ![image](https://github.com/user-attachments/assets/aeefe70a-bbfc-4467-a0e5-27628f94bfac)
   ![image](https://github.com/user-attachments/assets/bd3271c5-b0b5-46f4-9d48-90e66185054b)

   - Find Consecutive Zeroes Function<br/>
    ![image](https://github.com/user-attachments/assets/c15b952c-08e8-43c5-9b7a-1a68b557b666)

   - Extract Magnet Mask Function<br/>
   ![image](https://github.com/user-attachments/assets/b929b9eb-1666-4814-b702-70652c088179)
    ![image](https://github.com/user-attachments/assets/77e27dbd-18af-473d-a05b-8eff3225a804)

   - Magnet CUDA Function<br/>
   ![image](https://github.com/user-attachments/assets/70bf328a-cd0a-47d3-802f-e7de76847935)
    ![image](https://github.com/user-attachments/assets/8cde2d75-a218-4d6a-9a56-5c5217533b31)

 - Correctness checker<br/>
  ![image](https://github.com/user-attachments/assets/5c22772d-6d6e-43c1-bba1-249d0c35f0f1)


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


- **Problems encountered** <br/>
Race conditions occurred with the threads in a block when writing in the shared memory. This caused incorrect values to be added to each histogram bin which produced the incorrect results. Multiple threads writing to the same memory location causing incorrect incrementation of some bins resulting in incorrect results.

- **AHA moments** <br/>
The use of the __syncthreads() function was implemented to allow for all threads in a block to finish updating the shared memory before merging with the global memory. The additional use of atomicAdd() to prevent two or more threads from updating the same memory location, preventing incorrect results.
