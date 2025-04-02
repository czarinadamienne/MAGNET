# MAGNET
# Group 6 - Integrating Project
## YOUTUBE LINK: 

- This project aimts to implement the MAGNET DNA pre-alignment filter in C and utilize CUDA programming to parallelize certain functions in the source code. The CUDA implementation uses shared memory concept, memory management and atomic operations. 
  
## Members

* Alamay, Carl Justine
* Ang, Czarina Damienne
* Esteban, Janina Angela
* Herrera, Diego Martin

## Report

### I. Screenshot of the program output with execution time and correctness check (C)
   ![image](https://github.com/user-attachments/assets/27c8ffd3-09e1-400d-9d5d-3e0b77011ff0)
<br/>
### II. Screenshot of the program output with execution time and correctness check (CUDA)
**256 Threads**
  ![image](https://github.com/user-attachments/assets/06dd69b1-420a-4316-a5d2-02386bac75e8)
<br/>
**1024 Threads**
  ![image](https://github.com/user-attachments/assets/4f736d28-b66d-4a27-b68c-3172b735b43d)
<br/>
### III. Screenshot of the program implementation
**1. C Implementation**
   - Histogram Count function<br/>
     ![image](https://github.com/user-attachments/assets/c87c1404-1a1d-4206-9a4b-7f360cb195b3)
   - Correctness checker<br/>
     ![image](https://github.com/user-attachments/assets/e09484fe-b7d0-4469-8028-263fb27231c6)

  As shown above, the function computes for the index by looping through the vector and getting its remainder when divided by 10. The computed index is used to locate the specific histogram to increment.
**2. CUDA Implementation**
   - Histogram Count kernel<br/>
     ![image](https://github.com/user-attachments/assets/ee82de7f-76ff-42c1-a329-34b1cd1511ad)
   - Correctness checker<br/>
     ![image](https://github.com/user-attachments/assets/57496315-5671-4d53-8f43-695ee7ecfb96)

     Similar to the C implementation, but this applied shared memory. A static shared memory called sharedHist with 10 elements is initialized. The sharedHist is first initialized to 0 to properly increment the values inside it, later on. __syncthreads() is called after to synchronize the threads since each threads are executing in parallel in a block. This is to avoid race condition. The next block computes for the index of the histogram to be incremented. atomicAdd() is used to read the address in shared memory, adds one, and writes the result back to the same address read. When this is called, no threads can access the same address until writing of the results is finished. To write back to global memory, atomicAdd is also used. The kernel only modifies the values first in the shared memory and then the result will be written back to the global memory, histbins[].
     
### IV. Comparative table of execution time and Analysis
Average execution time of C Program
| Vector size | C Program      | 
| ----------- | -------------- | 
| 1 << 28     | 1667.850767 ms |


Average execution times of CUDA
| Vector size = 1 << 28 | CUDA Program      | Speedup compared to C |
| --------------------- | ----------------- | --------------------- |
| Threads = 256         | 96.590928 ms      | 17.3x                 |
| Threads = 1024        | 98.698222 ms      | 16.9x                 |


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

- **When is it faster/better to use shared memory?** <br/>
It is better to use shared memory when we have to repeatedly access and modify the same data. It is also better to use this when manupulating large datasets. Moreover, it is also faster because it accesses and modifies local data or the data that is already inside the GPU chip, compared to C, wherein it has to call or pass the histogram bins outside of the function. 
