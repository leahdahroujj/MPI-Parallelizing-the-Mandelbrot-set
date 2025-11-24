#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

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

int main(int argc, char** argv) {
    int rank, size;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int rows_per_process = HEIGHT / size;
    int remainder = HEIGHT % size;
    
    int start_row, end_row, num_rows;
    if (rank < remainder) {
        start_row = rank * (rows_per_process + 1);
        num_rows = rows_per_process + 1;
    } else {
        start_row = rank * rows_per_process + remainder;
        num_rows = rows_per_process;
    }
    end_row = start_row + num_rows;
    
    int *local_image = (int*)malloc(num_rows * WIDTH * sizeof(int));
    if (local_image == NULL) {
        printf("Error: Process %d failed to allocate memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    start_time = MPI_Wtime();
    

    struct complex c;
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < WIDTH; j++) {
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
            local_image[(i - start_row) * WIDTH + j] = cal_pixel(c);
        }
    }
    
    end_time = MPI_Wtime();
    
    int *image = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        image = (int*)malloc(HEIGHT * WIDTH * sizeof(int));
        if (image == NULL) {
            printf("Error: Root process failed to allocate image memory\n");
            free(local_image);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        if (recvcounts == NULL || displs == NULL) {
            printf("Error: Root process failed to allocate gather arrays\n");
            free(local_image);
            free(image);
            if (recvcounts) free(recvcounts);
            if (displs) free(displs);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        for (int i = 0; i < size; i++) {
            if (i < remainder) {
                recvcounts[i] = (rows_per_process + 1) * WIDTH;
                displs[i] = i * (rows_per_process + 1) * WIDTH;
            } else {
                recvcounts[i] = rows_per_process * WIDTH;
                displs[i] = (i * rows_per_process + remainder) * WIDTH;
            }
        }
    }
    
    MPI_Gatherv(local_image, num_rows * WIDTH, MPI_INT,
                image, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    

    double local_time = end_time - start_time;
    double max_time, min_time, avg_time;
    
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        avg_time /= size;
        printf("=== Static Scheduling Results ===\n");
        printf("Number of processes: %d\n", size);
        printf("Image size: %d x %d\n", WIDTH, HEIGHT);
        printf("Max computation time: %f seconds\n", max_time);
        printf("Min computation time: %f seconds\n", min_time);
        printf("Avg computation time: %f seconds\n", avg_time);
        printf("Total parallel time: %f seconds\n", max_time);
        
        save_pgm("images/mandelbrot_static.pgm", image);
        printf("Image saved to images/mandelbrot_static.pgm\n");
        
        free(image);
        free(recvcounts);
        free(displs);
    }
    
    free(local_image);
    MPI_Finalize();
    return 0;
}
