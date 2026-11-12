#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

typedef struct {
    TSparseMatrix *A;
    TSparseMatrix *B;

    int *a_first;  
    int *a_next;   
    int *b_first;  
    int *b_next;  

    unsigned int next_row;
    size_t capacity;
    int failed;
    TSparseMatrix *C;
    pthread_mutex_t lock;
} Shared;

static int add_result(Shared *s, int row, int col, double value)
{
    if (value == 0.0) return 1;

    if (s->C->nnz == UINT_MAX) return 0;

    if (s->C->nnz == s->capacity) {
        size_t new_capacity = s->capacity ? s->capacity * 2 : 16;

        if (new_capacity <= s->capacity ||
            new_capacity > SIZE_MAX / sizeof(COOElement)) {
            return 0;
        }

        COOElement *new_elements =
            realloc(s->C->elements, new_capacity * sizeof(COOElement));

        if (new_elements == NULL) return 0;

        s->C->elements = new_elements;
        s->capacity = new_capacity;
    }

    COOElement *entry = &s->C->elements[s->C->nnz++];
    entry->r = row;
    entry->c = col;
    entry->val = value;
    return 1;
}

static void *work_on_rows(void *argument)
{
    Shared *s = argument;

    double *sum = calloc(s->B->ncols, sizeof(double));
    unsigned char *used = calloc(s->B->ncols, sizeof(unsigned char));
    int *touched = malloc((size_t)s->B->ncols * sizeof(int));

    if (sum == NULL || used == NULL || touched == NULL) {
        pthread_mutex_lock(&s->lock);
        s->failed = 1;
        pthread_mutex_unlock(&s->lock);
        free(sum);
        free(used);
        free(touched);
        return NULL;
    }

    for (;;) {
        unsigned int row;

        pthread_mutex_lock(&s->lock);
        if (s->failed || s->next_row == s->A->nrows) {
            pthread_mutex_unlock(&s->lock);
            break;
        }
        row = s->next_row++;
        pthread_mutex_unlock(&s->lock);

        int touched_count = 0;

        for (int ai = s->a_first[row]; ai != -1; ai = s->a_next[ai]) {
            COOElement a = s->A->elements[ai];
            int k = a.c;

            for (int bi = s->b_first[k]; bi != -1; bi = s->b_next[bi]) {
                COOElement b = s->B->elements[bi];
                int col = b.c;

                if (!used[col]) {
                    used[col] = 1;
                    touched[touched_count++] = col;
                }
                sum[col] += a.val * b.val;
            }
        }

        pthread_mutex_lock(&s->lock);

        if (!s->failed) {
            for (int i = 0; i < touched_count; ++i) {
                int col = touched[i];
                if (!add_result(s, (int)row, col, sum[col])) {
                    s->failed = 1;
                    break;
                }
            }
        }

        pthread_mutex_unlock(&s->lock);

        for (int i = 0; i < touched_count; ++i) {
            int col = touched[i];
            sum[col] = 0.0;
            used[col] = 0;
        }
    }

    free(sum);
    free(used);
    free(touched);
    return NULL;
}

TSparseMatrix *spmm_parallel(TSparseMatrix *A, TSparseMatrix *B,
                             int num_threads)
{
    if (A == NULL || B == NULL || num_threads <= 0 ||
        A->ncols != B->nrows ||
        (A->nnz && A->elements == NULL) ||
        (B->nnz && B->elements == NULL) ||
        A->nnz > INT_MAX || B->nnz > INT_MAX ||
        A->nrows > INT_MAX || B->ncols > INT_MAX) {
        return NULL;
    }

    Shared s = {0};
    s.A = A;
    s.B = B;

    s.C = malloc(sizeof(TSparseMatrix));
    if (s.C == NULL) return NULL;

    s.C->nrows = A->nrows;
    s.C->ncols = B->ncols;
    s.C->nnz = 0;
    s.C->elements = NULL;

    if (A->nrows == 0 || B->ncols == 0 ||
        A->nnz == 0 || B->nnz == 0) {
        return s.C;
    }

    s.a_first = malloc((size_t)A->nrows * sizeof(int));
    s.a_next = malloc((size_t)A->nnz * sizeof(int));
    s.b_first = malloc((size_t)B->nrows * sizeof(int));
    s.b_next = malloc((size_t)B->nnz * sizeof(int));

    if (!s.a_first || !s.a_next || !s.b_first || !s.b_next)
        goto failure;

    for (unsigned int r = 0; r < A->nrows; ++r) s.a_first[r] = -1;
    for (unsigned int r = 0; r < B->nrows; ++r) s.b_first[r] = -1;

    for (unsigned int i = 0; i < A->nnz; ++i) {
        int r = A->elements[i].r;
        int c = A->elements[i].c;
        if (r < 0 || c < 0 ||
            (unsigned int)r >= A->nrows ||
            (unsigned int)c >= A->ncols) goto failure;

        s.a_next[i] = s.a_first[r];
        s.a_first[r] = (int)i;
    }

    for (unsigned int i = 0; i < B->nnz; ++i) {
        int r = B->elements[i].r;
        int c = B->elements[i].c;
        if (r < 0 || c < 0 ||
            (unsigned int)r >= B->nrows ||
            (unsigned int)c >= B->ncols) goto failure;

        s.b_next[i] = s.b_first[r];
        s.b_first[r] = (int)i;
    }

    if (pthread_mutex_init(&s.lock, NULL) != 0) goto failure;

    unsigned int thread_count = (unsigned int)num_threads;
    if (thread_count > A->nrows) thread_count = A->nrows;

    pthread_t *threads = malloc((size_t)thread_count * sizeof(pthread_t));
    if (threads == NULL) {
        pthread_mutex_destroy(&s.lock);
        goto failure;
    }

    unsigned int started = 0;
    for (unsigned int i = 0; i < thread_count; ++i) {
        if (pthread_create(&threads[i], NULL, work_on_rows, &s) != 0) {
            pthread_mutex_lock(&s.lock);
            s.failed = 1;
            pthread_mutex_unlock(&s.lock);
            break;
        }
        started++;
    }

    for (unsigned int i = 0; i < started; ++i)
        pthread_join(threads[i], NULL);

    free(threads);
    pthread_mutex_destroy(&s.lock);

    if (s.failed) goto failure;

    free(s.a_first);
    free(s.a_next);
    free(s.b_first);
    free(s.b_next);
    return s.C;

failure:
    free(s.a_first);
    free(s.a_next);
    free(s.b_first);
    free(s.b_next);
    free(s.C->elements);
    free(s.C);
    return NULL;
}
