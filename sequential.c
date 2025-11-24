#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex{
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
    }
    while ((iter < MAX_ITER) && (lengthsq < 4.0));
    
    return iter;
}

void save_pgm(const char *filename, int **image) {
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
            temp = image[i][j]; 
            fprintf(pgmimg, "%d ", temp);
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 
} 


int main() {
    int **image = (int **)malloc(HEIGHT * sizeof(int *));
    if (image == NULL) {
        printf("Error: Memory allocation failed for image rows\n");
        return 1;
    }
    
    for (int i = 0; i < HEIGHT; i++) {
        image[i] = (int *)malloc(WIDTH * sizeof(int));
        if (image[i] == NULL) {
            printf("Error: Memory allocation failed for image row %d\n", i);

            for (int j = 0; j < i; j++) {
                free(image[j]);
            }
            free(image);
            return 1;
        }
    }
    
    double AVG = 0;
    int N = 10;
    double *total_time = (double *)malloc(N * sizeof(double));
    
    if (total_time == NULL) {
        printf("Error: Memory allocation failed for timing array\n");
        for (int i = 0; i < HEIGHT; i++) {
            free(image[i]);
        }
        free(image);
        return 1;
    }
    
    struct complex c;

    for (int k = 0; k < N; k++) {
        clock_t start_time = clock();
        
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[i][j] = cal_pixel(c);
            }
        }

        clock_t end_time = clock();
        total_time[k] = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Execution time of trial [%d]: %f seconds\n", k + 1, total_time[k]);
        AVG += total_time[k];
    }

    save_pgm("images/mandelbrot_seq.pgm", image);
    printf("The average execution time of 10 trials is: %f ms\n", AVG / N * 1000);

    for (int i = 0; i < HEIGHT; i++) {
        free(image[i]);
    }
    free(image);
    free(total_time);

    return 0;
}
