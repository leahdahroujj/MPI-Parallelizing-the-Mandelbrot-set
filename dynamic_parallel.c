#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255
#define MASTER 0
#define WORK_TAG 1
#define RESULT_TAG 2
#define TERMINATE_TAG 3

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
    
    return iter;
}

void save_pgm(const char *filename, int *image) {
    FILE* pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    
    if (pgmimg == NULL) {
        printf("Error: Cannot create file %s\n", filename);
        return;
    }
    
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");
    
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i * WIDTH + j];
            fprintf(pgmimg, "%d ", temp);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

void master_process(int size, int *image) {
    int row_num = 0;
    int completed_rows = 0;
    int worker_id;
    MPI_Status status;
    

    int *row_data = (int *)malloc(WIDTH * sizeof(int));
    if (row_data == NULL) {
        printf("Error: Master failed to allocate row buffer\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    

    for (worker_id = 1; worker_id < size && row_num < HEIGHT; worker_id++) {
        MPI_Send(&row_num, 1, MPI_INT, worker_id, WORK_TAG, MPI_COMM_WORLD);
        row_num++;
    }
    

    while (completed_rows < HEIGHT) {

        MPI_Recv(row_data, WIDTH, MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, 
                 MPI_COMM_WORLD, &status);
        worker_id = status.MPI_SOURCE;
        

        int completed_row;
        MPI_Recv(&completed_row, 1, MPI_INT, worker_id, RESULT_TAG, 
                 MPI_COMM_WORLD, &status);
        

        for (int j = 0; j < WIDTH; j++) {
            image[completed_row * WIDTH + j] = row_data[j];
        }
        completed_rows++;
        

        if (row_num < HEIGHT) {
            MPI_Send(&row_num, 1, MPI_INT, worker_id, WORK_TAG, MPI_COMM_WORLD);
            row_num++;
        } else {
            int terminate = -1;
            MPI_Send(&terminate, 1, MPI_INT, worker_id, TERMINATE_TAG, MPI_COMM_WORLD);
        }
    }
    
    free(row_data);
}

void worker_process(int rank) {
    MPI_Status status;
    int row_num;
    struct complex c;
    

    int *row_data = (int *)malloc(WIDTH * sizeof(int));
    if (row_data == NULL) {
        printf("Error: Worker %d failed to allocate row buffer\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    while (1) {
        MPI_Recv(&row_num, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        if (status.MPI_TAG == TERMINATE_TAG) {
            break;
        }
        

        for (int j = 0; j < WIDTH; j++) {
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (row_num - HEIGHT / 2.0) * 4.0 / HEIGHT;
            row_data[j] = cal_pixel(c);
        }
        

        MPI_Send(row_data, WIDTH, MPI_INT, MASTER, RESULT_TAG, MPI_COMM_WORLD);
        MPI_Send(&row_num, 1, MPI_INT, MASTER, RESULT_TAG, MPI_COMM_WORLD);
    }
    
    free(row_data);
}

int main(int argc, char** argv) {
    int rank, size;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) {
        if (rank == 0) {
            printf("Error: This program requires at least 2 processes (1 master + 1 worker)\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    start_time = MPI_Wtime();
    
    if (rank == MASTER) {
        int *image = (int *)malloc(HEIGHT * WIDTH * sizeof(int));
        if (image == NULL) {
            printf("Error: Master failed to allocate image memory\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        master_process(size, image);
        
        end_time = MPI_Wtime();
        
        printf("=== Dynamic Scheduling Results ===\n");
        printf("Number of processes: %d (1 master + %d workers)\n", size, size - 1);
        printf("Image size: %d x %d\n", WIDTH, HEIGHT);
        printf("Total parallel time: %f seconds\n", end_time - start_time);
        
        save_pgm("images/mandelbrot_dynamic.pgm", image);
        printf("Image saved to images/mandelbrot_dynamic.pgm\n");
        
        free(image);
    } else {
        worker_process(rank);
        end_time = MPI_Wtime();
        printf("Worker %d computation time: %f seconds\n", rank, end_time - start_time);
    }
    
    MPI_Finalize();
    return 0;
}
