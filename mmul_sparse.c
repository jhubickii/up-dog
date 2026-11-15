#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "matrix.h"
#include "mmio.h"
#include <limits.h>
#include <stdint.h>

/* Ingestion engine utilizing the official NIST Matrix Market Library */
TSparseMatrix *read_coo_matrix(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Failed to open matrix file");
        return NULL;
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        fclose(f);
        return NULL;
    }

    // Validation rule: verify format is real, coordinate, general matrix
    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode) || !mm_is_real(matcode)) {
        fprintf(stderr, "Unsupported matrix type. Must be a real, coordinate matrix.\n");
        fclose(f);
        return NULL;
    }

    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "Failed to parse matrix dimension size tokens.\n");
        fclose(f);
        return NULL;
    }

    TSparseMatrix *matrix = malloc(sizeof(TSparseMatrix));
    if (!matrix) return NULL;
    
    matrix->nrows = M;
    matrix->ncols = N;
    matrix->nnz = nz;
    matrix->elements = malloc(nz * sizeof(COOElement));
    if (!matrix->elements) {
        free(matrix);
        fclose(f);
        return NULL;
    }

    for (int i = 0; i < nz; i++) {
        int row, col;
        double value;
        // Matrix Market format is 1-indexed; convert down to standard 0-indexed structures
        if (fscanf(f, "%d %d %lg\n", &row, &col, &value) != 3) {
            fprintf(stderr, "Error parsing element line %d\n", i);
            free(matrix->elements);
            free(matrix);
            fclose(f);
            return NULL;
        }
        matrix->elements[i].r = row - 1;
        matrix->elements[i].c = col - 1;
        matrix->elements[i].val = value;
    }

    fclose(f);
    return matrix;
}

void free_sparse_matrix(TSparseMatrix *m) {
    if (m) {
        if (m->elements) free(m->elements);
        free(m);
    }
}

---------------------------------------------------------------------------------------------------------------------------
static int cmp_coo(const void *pa, const void *pb)
{
    const COOElement *a = pa;
    const COOElement *b = pb;

    if (a->r != b->r) return (a->r > b->r) - (a->r < b->r);
    return (a->c > b->c) - (a->c < b->c);
}

/* TODO: Implement sequential SpMM baseline */
TSparseMatrix *spmm_sequential(TSparseMatrix *A, TSparseMatrix *B) {
    if (A == NULL || B == NULL || A->ncols != B->nrows) return NULL;
    if ((A->nnz && A->elements == NULL) || (B->nnz && B->elements == NULL)) {
        return NULL;
    }
    if (A->nnz > INT_MAX || B->nnz > INT_MAX ||
        A->nrows > INT_MAX || B->ncols > INT_MAX) {
        return NULL;
    }

    TSparseMatrix *C = malloc(sizeof(TSparseMatrix));
    if (C == NULL) return NULL;

    C->nrows = A->nrows;
    C->ncols = B->ncols;
    C->nnz = 0;
    C->elements = NULL;

    if (A->nrows == 0 || B->ncols == 0 || A->nnz == 0 || B->nnz == 0) {
        return C;
    }

    size_t *b_start = NULL;   /* b_start[k]..b_start[k+1] = entries of B in row k */
    size_t *cursor = NULL;
    COOElement *b_sorted = NULL;
    COOElement *prod = NULL;
    size_t total = 0;
    size_t out = 0;
    unsigned int i;

    b_start = calloc((size_t)B->nrows + 1, sizeof(size_t));
    cursor = malloc((size_t)B->nrows * sizeof(size_t));
    b_sorted = malloc((size_t)B->nnz * sizeof(COOElement));
    if (!b_start || !cursor || !b_sorted) goto fail;

    /* Validate indices and bucket B's entries by row (counting sort). */
    for (i = 0; i < B->nnz; ++i) {
        int r = B->elements[i].r;
        int c = B->elements[i].c;
        if (r < 0 || c < 0 ||
            (unsigned int)r >= B->nrows || (unsigned int)c >= B->ncols) {
            goto fail;
        }
        b_start[r + 1]++;
    }
    for (unsigned int k = 0; k < B->nrows; ++k) {
        b_start[k + 1] += b_start[k];
    }
    for (unsigned int k = 0; k < B->nrows; ++k) {
        cursor[k] = b_start[k];
    }
    for (i = 0; i < B->nnz; ++i) {
        b_sorted[cursor[B->elements[i].r]++] = B->elements[i];
    }

    /* Validate A and count the exact number of products. */
    for (i = 0; i < A->nnz; ++i) {
        int r = A->elements[i].r;
        int k = A->elements[i].c;
        if (r < 0 || k < 0 ||
            (unsigned int)r >= A->nrows || (unsigned int)k >= A->ncols) {
            goto fail;
        }
        size_t cnt = b_start[k + 1] - b_start[k];
        if (cnt > SIZE_MAX - total) goto fail;
        total += cnt;
    }

    if (total == 0) {
        goto done;
    }
    if (total > SIZE_MAX / sizeof(COOElement)) goto fail;

    prod = malloc(total * sizeof(COOElement));
    if (prod == NULL) goto fail;

    /* Generate every product A(i,k) * B(k,j) as a triplet (i, j, value). */
    size_t w = 0;
    for (i = 0; i < A->nnz; ++i) {
        COOElement a = A->elements[i];
        size_t end = b_start[a.c + 1];
        for (size_t p = b_start[a.c]; p < end; ++p) {
            prod[w].r = a.r;
            prod[w].c = b_sorted[p].c;
            prod[w].val = a.val * b_sorted[p].val;
            ++w;
        }
    }

    /* Group equal (row, col) keys together. */
    qsort(prod, total, sizeof(COOElement), cmp_coo);

    /* Sum each run of equal keys and compact the result in place.
       The write index never passes the read index, so this is safe. */
    for (size_t p = 0; p < total; ) {
        int r = prod[p].r;
        int c = prod[p].c;
        double sum = 0.0;

        while (p < total && prod[p].r == r && prod[p].c == c) {
            sum += prod[p].val;
            ++p;
        }

        if (sum != 0.0) {
            if (out == UINT_MAX) goto fail;
            prod[out].r = r;
            prod[out].c = c;
            prod[out].val = sum;
            ++out;
        }
    }

done:
    free(b_start);
    free(cursor);
    free(b_sorted);

    if (out == 0) {
        free(prod);
        prod = NULL;
    } else {
        /* Shrink to the actual size. If realloc fails, the original block
           is still valid and can be used as is. */
        COOElement *shrunk = realloc(prod, out * sizeof(COOElement));
        if (shrunk != NULL) prod = shrunk;
    }

    C->elements = prod;
    C->nnz = (unsigned int)out;
    return C;

fail:
    free(b_start);
    free(cursor);
    free(b_sorted);
    free(prod);
    free(C);

    // Group workspace...
    
    return NULL; 
}
-----------------------------------------------------------------------------------------------------------------
/* TODO: Implement multi-threaded SpMM parallel architecture */
/*
TSparseMatrix *spmm_parallel(TSparseMatrix *A, TSparseMatrix *B, int num_threads) {
    if (A == NULL || B == NULL || A->ncols != B->nrows || num_threads <= 0) return NULL;

    // Group workspace...
    return NULL;
}
    */