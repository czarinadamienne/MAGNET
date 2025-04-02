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
   ![Untitled2](https://github.com/user-attachments/assets/4cc1a30a-6636-45b5-ae70-cf3aa7238228)

   **E = 3**
   ![Untitled](https://github.com/user-attachments/assets/9f1a1adf-c473-459b-b710-2ebc124df5f4)

   **E = 8**
   ![image](https://github.com/user-attachments/assets/ff44b3a5-b5d4-47be-8d56-6e2adaddbbea)

   
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

The Extraction_Encapsulation() function finds long sequences of zeros in a given range and marks in the MagnetMask. It starts by storing the range in a stack and processes it until all sections are checked. For each range, it looks at different versions of the sequence and finds the longest group of consecutive zeros. If a second-longest group exists, it keeps track of that too. Once the longest zero sequence is found, it marks those positions as zero in MagnetMask. The function will then split the remaining unprocessed parts and add them back to the stack to continue searching.
 
   - MAGNET function<br/>
    ![image](https://github.com/user-attachments/assets/326bf040-8377-4b99-888f-31c1da944611)

  ![image](https://github.com/user-attachments/assets/1e60b933-3afb-4a2c-99dc-09a8ea446b9d)

  ![image](https://github.com/user-attachments/assets/43fac034-12d8-47de-845f-fee05d5bbff8)

The MAGNET() function compares the reference and read sequence to assess whether they are within the predefined error threshold. It first creates hamming masks that mark differences between the sequences. If the number of differences is equal or less than the error threshold, then the function immediately accepts the sequence. Otherwise, it creates extra hamming masks by shifting the read sequence left and right, accounting for possible insertions or deletions.

The function will then call Extraction_Encapsulation() to identify the longest consecutive streaks of zeros present within the insertion and deletion masks. These zero streaks represent regions where the read sequence aligns well with the reference. The function updates the MagnetMask to mark these high confidence regions while filtering out misaligned ones. The remaining mismatches will then be counted, and if it's more than the predefined edit threshold, then it will be rejected; Otherwise, it will be accepted.
    
   - Correctness checker<br/>
  ![image](https://github.com/user-attachments/assets/954d3b87-2b06-4825-81f8-f0c1fb61939f)

The snippet of the code showing the correctness checker compares two bit vectors, those being the finalBitVector from the intended output and another from a comparison variant. It iterates over the entire sequence being compared, and for every index, it checks if the corresponding bit in both the bit vectors match or not. If they are a mismatch at that specific index, then the error counter will be incremented by 1. This is to keep track of the number of matches between the two bit vectors.   

The MAGNET pre-alignment filter would first create a hamming mask of the query and reference sequence through checking the nucleobase of the query and reference sequence in the same location. It would mark as 0 if the sequences matches, and 1 if it does not match. It would also create a deletion and insertion mask, having a total of 2E+1 hamming masks, where E is the edit distance threshold. This parameter could be set by the user. To create a deletion mask, it would shift the query sequence to the right, and to create an insertion mask, it would shift the query to the left. The number of shifts are dependent on the edit distance threshold. This is done in the MAGNET function. Next, it would find the longest consecutive zeroes in each masks and its information such as number of zeroes, and start and end index of the zeroes would be stored. The longest zeroes for each mask would be padded with 1s. This is done in Consecutive Zeroes function. It would iteratively use divide and conquer to find the longest zeroes among all the hamming masks and copy these zeroes to the final bit-vector. This is done in Extraction Encapsulation function. Lastly, it will count the number of 1s in the final bit-vector, if the number of 1s are greater than the edit distance threshold, the sequence will be rejected, but if the number of 1s are less than or equal to the edit distance threshold, the sequence will be accepted. This is done in the MAGNET function. 

  
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

text

### IV. Comparative table of execution time and Analysis
Average execution time of C Program
| Dataset size | E = 0      | E = 3 | E = 8 |
| ------------ | ---------- | ----- | ------ |
| 1000         |  12.36 ms | 79.94 ms | 189.54 ms |


Average execution times of CUDA
| Dataset size = 1000 | E = 0 | E = 3| E = 8 | Speedup compared to C |
| ------------------- | ----- |------|--------| --------------------- |
| Blocks = 256        |   ms |    ms |       ms  |      x  |
| Blocks = 1024       |   ms |    ms |        ms |       x  |


Both kernels were timed with a vector size of 2^28 and their average execution times after 30 loops of the function were compared. The C kernel had an average execution time of 1667.850767 ms while the CUDA kernel with 1024 threads had an average execution time of 98.698222 ms. The CUDA kernel with 1024 threads was around 16.9 times faster than the C kernel. Additionally, the CUDA kernel with 256 threads had an execution time of 96.590928 ms, which is a 17.3 times faster than the C kernel. It is also observed that 256 threads is faster, for this case, than 1024 threads. This could be because 256 threads is the enough amount of threads needed as there is less memory access and fewer conflicts, since shared memory is applied in the code. There could be bank conflicts if there are too many threads per block.
<br/>

The C kernel is generally slower in execution due to the sequential execution of the function during the loop. The CPU will have to iterate 2^28 elements one at a time. The CUDA kernel deploys multiple threads for the computation of the histogram bins to allow for parallel execution of the function. The use of unified memory, page creation, and mem advise also aided in the reduction of execution time for the CUDA kernel. The unified memory removes the manual copying of data between host and device, page creation to reduce the number of page faults that occur, and mem advise to only send the output of the GPU back to the CPU rather than both the input and the output. These three techniques lowers the overall overhead of the kernel and allows for faster execution time.
<br/>

The shared memory concept was also applied in the creation of the CUDA program. Instead of each thread block having to access the global memory, it instead accesses its own shared memory which stores frequently accessed data. Each thread will update its own copy in their own shared memory block which is later merged with the global memory later on. This removes the latency that comes with repeatedly accessing global memory which is not present when accessing shared memory. This further reduced the memory overhead that the CUDA kernel experiences.
<br/>

### V. Discussion

- **Parallel Algorithms** <br/>
The functionalities that were converted from sequential to parallel processing were the hamming distance computation, extraction and encapsulation function, and the locating of the longest set of consecutive zeros in a mask.

The hamming mask calculation is used to track the number of errors between the reference and query genome. The hamming mask calculation for the sequential processing implementation was originally placed into the MAGNET function. One corresponding character from the reference and query genome were compared one at a time during execution. This process was converted into a parallel algorithm through the creation of the computeHammingAndErrorsKernel function. Instead of comparing sequentially, multiple threads are deployed and each one is assigned to compare one corresponding character from the reference and query genome which allows multiple comparisons to be made in one cycle. Global memory is then used to store the results from these comparisons with the additional use of atomic operations to prevent race conditions from occurring. 

The extraction and encapsulation function in the original sequential implementation uses a stack-based approach to modify the segment runs of zeros in the hamming masks. This process is done by processing one sequence region at a time. In the parallelized version, multiple threads are assigned to process more sequence regions at a time and update these hamming masks in parallel. 

The searching for consecutive zeros is done sequentially by iteratively looking over each character in the hamming mask. It marks the starting index of the zero sequence and marks the end index when looking at a non-zero number. This does it one sequence at a time and loops until all sequences have been checked. For the parallel implementation, multiple threads process different portions of the arrays with the additional use of global memory for storage of results while using atomic operations to avoid race conditions.

- **Problems encountered** <br/>
One problem encountered during the creation of the project was the conversion of the MATLAB source code into the C code. The original implementation had the functions separated so unifying them into one source code proved difficult. Another problem encountered was the encountering of segmentation faults at higher dataset sizes for the CUDA implementation. It is also observed as the Edit distance threshold increases, there would be a difference of 2 in between the number of accepted sequences in CUDA and C. 

- **AHA moments** <br/>
The processes that were best suited for parallelization in the original sequential implementation were the processes that utilized iterative statements to process their data. Since sequential processing processes one piece of data per cycle, these areas proved the best suited for parallelization. 
