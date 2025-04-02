#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#define MAX_SEQ_LEN 1024
#define MAX_SEQUENCES 1000
#define STACK_SIZE 1000

typedef struct
{
    int* lengths;
    int* starts;
    int* ends;
    int count;
} ConsecutiveZerosInfo;

typedef struct{
  int start;
  int end;
} Range;

ConsecutiveZerosInfo ConsecutiveZeros(int* sequence, int length);
void Extraction_Encapsulation(int** HammingMask, int start, int end, int* MagnetMask, int ErrorThreshold, int read_len, int* accepted);
int MAGNET(const char* RefSeq, const char* ReadSeq, int ErrorThreshold, int* finalBitVector, int** HammingMask, int* accepted);
int readCSVFile(char sequences[MAX_SEQUENCES][MAX_SEQ_LEN], char refSequences[MAX_SEQUENCES][MAX_SEQ_LEN], int editThresholds[MAX_SEQUENCES], int *numSequences);
void printStats(int acceptedCount, int rejectedCount, int errors, int numSequences, double elapse);

void printStats(int acceptedCount, int rejectedCount, int errors, int numSequences, double elapse)
{
  printf("----- Statistics -----\n\n");
  printf("Number of accepted sequences: %d\n", acceptedCount);
  printf("Number of rejected sequences: %d\n", rejectedCount);
  printf("Number of sequences processed: %d\n", numSequences);
  printf("Number of runs: 10\n");
  printf("Average time taken: %f microseconds\n", elapse / 10);
  printf("Number of errors: %d\n", errors);
}

ConsecutiveZerosInfo ConsecutiveZeros(int* sequence, int length)
{
    int* starts = (int*)malloc(length * sizeof(int));
    int* ends = (int*)malloc(length * sizeof(int));
    int* lengths = (int*)malloc(length * sizeof(int));

    int count = 0;
    int in_zero_run = 0;
    int start_idx = 0;

    for (int i = 0; i < length; i++)
    {
        if (sequence[i] == 0)
        {
            if (!in_zero_run)
            {
                start_idx = i;
                in_zero_run = 1;
            }
        }
        else
        {
            if (in_zero_run)
            {
                ends[count] = i - 1;
                starts[count] = start_idx;
                lengths[count] = ends[count] - starts[count] + 1;
                count++;
                in_zero_run = 0;
            }
        }
    }

    if (in_zero_run)
    {
        ends[count] = length - 1;
        starts[count] = start_idx;
        lengths[count] = ends[count] - starts[count] + 1;
        count++;
    }

    return (ConsecutiveZerosInfo){lengths, starts, ends, count};
}

void Extraction_Encapsulation(int** HammingMask, int start, int end, int* MagnetMask, int ErrorThreshold, int read_len, int* accepted) {
    Range stack[STACK_SIZE];
    int top = -1;

    stack[++top] = (Range){start, end};

    while (top >= 0) {
        Range current = stack[top--];
        start = current.start;
        end = current.end;

        if (start > end) continue;

        int max_run = 0, max_start = -1, max_end = -1;
        int second_max_start = -1, second_max_end = -1;

        for (int mask_id = 0; mask_id <= 2 * ErrorThreshold; mask_id++) {
            int sub_len = end - start + 1;
            int* sub_seq = (int*)malloc(sub_len * sizeof(int));

            for (int i = 0; i < sub_len; i++) {
                sub_seq[i] = HammingMask[mask_id][start + i];
            }

            ConsecutiveZerosInfo info = ConsecutiveZeros(sub_seq, sub_len);
            free(sub_seq);

            for (int i = 0; i < info.count; i++) {
                if (info.lengths[i] > max_run) {
                    second_max_start = max_start;
                    second_max_end = max_end;

                    max_run = info.lengths[i];
                    max_start = start + info.starts[i];
                    max_end = start + info.ends[i];
                }
            }

            free(info.lengths);
            free(info.starts);
            free(info.ends);
        }

        if (max_run == 0 || max_start == -1 || max_end == -1) continue;

        for (int i = max_start; i <= max_end; i++) {
            MagnetMask[i] = 0;
        }

        if (second_max_start != -1 && second_max_end != -1) {
            if (second_max_start > max_end + 1) {
                MagnetMask[max_end + 1] = 1;
                MagnetMask[max_end + 2] = 1;
            }
        } else if (max_end + 1 < read_len) {
            MagnetMask[max_end + 1] = 1;
        }

        if (max_start - 2 >= start) {
            stack[++top] = (Range){start, max_start - 2};
        }

        if (max_end + 2 <= end) {
            stack[++top] = (Range){max_end + 2, end};
        }
    }
}

int MAGNET(const char* RefSeq, const char* ReadSeq, int ErrorThreshold, int* finalBitVector, int** HammingMask, int* accepted)
{
    int read_len = strlen(ReadSeq);
    int max_masks = 2 * ErrorThreshold + 1;
    *accepted = 0;

    // Initialize Hamming masks
    for (int i = 0; i < max_masks; i++)
    {
        HammingMask[i] = (int*)calloc(read_len, sizeof(int));
    }

    // Initialize MagnetMask to 1s
    int* MagnetMask = (int*)malloc(read_len * sizeof(int));
    for (int i = 0; i < read_len; i++) MagnetMask[i] = 1;

    int error_count = 0;
    for (int i = 0; i < read_len; i++)
    {
        HammingMask[0][i] = (ReadSeq[i] != RefSeq[i]);
        error_count += HammingMask[0][i];
    }

    if (error_count <= ErrorThreshold)
    {
        *accepted = 1;
        memcpy(finalBitVector, MagnetMask, read_len * sizeof(int));
        free(MagnetMask);
        return *accepted;
    }

    //Right-shifted masks (deletions)
    for (int e = 1; e <= ErrorThreshold; e++)
    {
        error_count = 0;
        memset(HammingMask[e], 0, read_len * sizeof(int));
        for (int i = e; i < read_len; i++)
        {
            if (i - e < strlen(ReadSeq))
            {
                HammingMask[e][i] = (ReadSeq[i - e] != RefSeq[i]);
                error_count += HammingMask[e][i];
            }
        }

        if (error_count <= ErrorThreshold)
        {
            *accepted = 1;
            memcpy(finalBitVector, MagnetMask, read_len * sizeof(int));
            free(MagnetMask);
            return *accepted;
        }
    }

    //Left-shifted masks (insertions)
    for (int e = 1; e <= ErrorThreshold; e++)
    {
        error_count = 0;
        memset(HammingMask[ErrorThreshold + e], 0, read_len * sizeof(int));
        for (int i = 0; i < read_len - e; i++)
        {
            if (i + e < strlen(ReadSeq))
            {
                HammingMask[ErrorThreshold + e][i] = (ReadSeq[i + e] != RefSeq[i]);
                error_count += HammingMask[ErrorThreshold + e][i];
            }
        }
        if (error_count <= ErrorThreshold)
        {
            *accepted = 1;
            memcpy(finalBitVector, MagnetMask, read_len * sizeof(int));
            free(MagnetMask);
            return *accepted;
        }
    }


    Extraction_Encapsulation(HammingMask, 0, read_len - 1, MagnetMask, ErrorThreshold, read_len, accepted);

    memcpy(finalBitVector, MagnetMask, read_len * sizeof(int));

    error_count = 0;
    for (int i = 0; i < read_len; i++)
    {
        error_count += MagnetMask[i];
    }

    *accepted = (error_count <= ErrorThreshold) ? 1 : 0;
    free(MagnetMask);
    return *accepted;
}

int readCSVFile(char sequences[MAX_SEQUENCES][MAX_SEQ_LEN], char refSequences[MAX_SEQUENCES][MAX_SEQ_LEN], int editThresholds[MAX_SEQUENCES], int *numSequences)
{
    FILE *file = fopen("dataset.csv", "r");
    if (!file)
    {
        printf("no file\n");
        return 0;
    }

    char line[MAX_SEQ_LEN];
    *numSequences = 0;
    while (fgets(line, sizeof(line), file) && *numSequences < MAX_SEQUENCES)
    {
        sscanf(line, "%[^,],%[^,],%d", sequences[*numSequences], refSequences[*numSequences], &editThresholds[*numSequences]);
        (*numSequences)++;
    }
    fclose(file);
    return 1;
}

int main()
{
    char (*sequences)[MAX_SEQ_LEN] = malloc(MAX_SEQUENCES * MAX_SEQ_LEN * sizeof(char));
    char (*refSequences)[MAX_SEQ_LEN] = malloc(MAX_SEQUENCES * MAX_SEQ_LEN * sizeof(char));
    int* editThresholds = malloc(MAX_SEQUENCES * sizeof(int));
    int numSequences;
    int accepted = 0;
    int acceptedC = 0;
    int errors = 0;
    int acceptedCount = 0;
    int rejectedCount = 0;
    int run = 10;
    double time = 0.0;
    double elapse = 0.0;
    int finalBitVector[MAX_SEQ_LEN];

    if (!readCSVFile(sequences, refSequences, editThresholds, &numSequences))
    {
        return 1;
    }

    for (int i = 0; i < numSequences; i++){
      int maskSize = 2 * editThresholds[i] + 1;
      int** HammingMask = (int**)malloc(maskSize * sizeof(int*));

      int cachemiss = MAGNET(refSequences[i], sequences[i], editThresholds[i], finalBitVector, HammingMask, &accepted);

      printf("Sequence %d:\nRead Sequence: %s\nReference Sequence: %s\nEdit Threshold: %d\n",
                i+1, sequences[i], refSequences[i], editThresholds[i]);

        for (int e = 0; e < 2 * editThresholds[i] + 1; e++)
        {
            printf("Mask %d: ", e);
            for (int j = 0; j < strlen(sequences[i]); j++) {
                printf("%d", HammingMask[e][j]);
            }
            printf("\n");
        }

        printf("Final Bit Vector: ");
        for (int j = 0; j < strlen(sequences[i]); j++)
        {
            printf("%d", finalBitVector[j]);
        }
        printf("\nResult: %s\n\n", cachemiss ? "Accepted" : "Rejected");
    }

    for (int j = 0; j < run; j++) {
        acceptedCount = 0;
        rejectedCount = 0;
        clock_t start, end;

        for (int i = 0; i < numSequences; i++) {
            int** HammingMask = (int**)malloc((2 * editThresholds[i] + 1) * sizeof(int*));

            for (int e = 0; e < 2 * editThresholds[i] + 1; e++)
            {
                HammingMask[e] = (int*)malloc(MAX_SEQ_LEN * sizeof(int));
                memset(HammingMask[e], 0, MAX_SEQ_LEN * sizeof(int));
            }

            //reset
            memset(finalBitVector, 0, sizeof(finalBitVector));
            int finalBitVectorC[MAX_SEQ_LEN];
            memset(finalBitVectorC, 0, sizeof(finalBitVectorC));

            start = clock();

            int result = MAGNET(refSequences[i], sequences[i], editThresholds[i], finalBitVector, HammingMask, &accepted);
            int checker = MAGNET(refSequences[i], sequences[i], editThresholds[i], finalBitVectorC, HammingMask, &acceptedC);

            end = clock();
            time = ((double)(end - start)) * 1E3 / CLOCKS_PER_SEC;
            elapse += time;

            if (result) {
                acceptedCount++;
            } else {
                rejectedCount++;
            }

            for (int k = 0; k < strlen(sequences[i]); k++) {
                if (finalBitVector[k] != finalBitVectorC[k]) {
                    errors++;
                }
            }

            // Free HammingMask memory
            for (int e = 0; e < 2 * editThresholds[i] + 1; e++) {
                free(HammingMask[e]);
            }
            free(HammingMask);
        }

        printf("Run %d Time: %.6f seconds | Accepted: %d | Rejected: %d\n",
               j + 1, time, acceptedCount, rejectedCount);
    }

    printf("\n");
    printStats(acceptedCount, rejectedCount, errors, numSequences, elapse);

    free(sequences);
    free(refSequences);
    free(editThresholds);

    return 0;
}