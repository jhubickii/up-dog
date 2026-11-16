/*
 * verify_tester.c — correctness and stress tester for the SpMM engines.
 *
 * What it does:
 *   1. Loads two Matrix Market files (A and B).
 *   2. Runs spmm_sequential once — this is the ground truth.
 *   3. Runs spmm_parallel with 1, 2, 4, 8, then 16 threads.
 *      Every parallel result is checked cell-by-cell against the
 *      sequential result using tolerance EPSILON = 1e-7.
 *   4. Times everything and prints a CSV summary you can paste into
 *      the report for speedup tables and charts.
 *
 * Usage:
 *   ./run-verification <MatrixA.mtx> <MatrixB.mtx> [repeats]
 *   repeats = runs per thread count (default 3). We keep the best time
 *   out of all repeats, since timing noise is always one-sided
 *   (interference can only make a run slower, never faster).
 *
 * Build note: read_coo_matrix, free_sparse_matrix, spmm_sequential and
 * spmm_parallel must be declared in matrix.h — they are.
 */

#define _POSIX_C_SOURCE 200809L /* needed for clock_gettime under -std=c99 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "matrix.h"

#define EPSILON 1e-7

/* The thread counts required by the assignment's stress-test mandate. */
static const int THREAD_COUNTS[] = { 1, 2, 4, 8, 16 };
#define NUM_THREAD_CONFIGS ((int)(sizeof(THREAD_COUNTS) / sizeof(THREAD_COUNTS[0])))
#define MAX_REPORTED_MISMATCHES 5

/* --------------------------------------------------------------- */
/* Timing                                                           */
/* --------------------------------------------------------------- */

/* Wall-clock timer. CLOCK_MONOTONIC never jumps backwards or follows
   NTP adjustments, unlike the plain system clock, so it is the right
   tool for measuring how long a computation took. */
static double now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* --------------------------------------------------------------- */
/* Hash map: (row, col) -> value                                    */
/* --------------------------------------------------------------- */

/*
 * Why a hash map?
 *
 * The obvious way to compare results is a nested loop: for every cell
 * in one result, scan the other result for the matching (row, col).
 * That is O(nnz^2) — with millions of non-zeros it would take hours.
 * With a hash map, inserting all cells of a matrix is O(nnz) and each
 * lookup is O(1) on average, so the whole comparison stays linear.
 *
 * Why SUM duplicate entries when inserting?
 *
 * COO output may contain several entries with the same (row, col)
 * (or the two engines may group/compact them differently). Summing
 * them while folding into the map collapses each logical cell to a
 * single value, so the comparison compares actual matrix contents,
 * not storage details.
 */

typedef struct {
    uint64_t key;   /* packed (row, col) */
    double  val;    /* accumulated sum for that cell */
    int     used;   /* 1 if this slot holds an entry */
} MapSlot;

typedef struct {
    MapSlot *slots;
    size_t   mask;  /* capacity - 1; capacity is always a power of two */
} CellMap;

/* Pack a cell coordinate into one 64-bit key.
   Safe because the spmm kernels reject dimensions above INT_MAX. */
static uint64_t cell_key(unsigned int r, unsigned int c)
{
    return ((uint64_t)r << 32) | (uint64_t)c;
}

/* Finalizer step of the SplitMix64 generator: scrambles the bits of
   the key so consecutive keys ((r,c) and (r,c+1)) spread evenly across
   the table instead of clustering. */
static uint64_t mix64(uint64_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

/* Create a table big enough for 'expected_entries' items.
   We size it at 2x the entry count so the table is at most half
   full — with open addressing, a half-empty table keeps probes short
   and guarantees a free slot is always reachable. */
static int cell_map_init(CellMap *m, size_t expected_entries)
{
    size_t cap = 16;

    if (expected_entries > (SIZE_MAX / 2) / sizeof(MapSlot)) return 0;
    while (cap < expected_entries * 2) cap <<= 1;

    m->slots = calloc(cap, sizeof(MapSlot)); /* zeroes 'used' flags */
    m->mask  = cap - 1;
    return m->slots != NULL;
}

/* Insert-or-add: if the key exists, add to its value; otherwise store
   the value under that key. Uses linear probing: if a slot is taken
   by a different key, just try the next slot. */
static void cell_map_add(CellMap *m, uint64_t key, double value)
{
    size_t i = (size_t)mix64(key) & m->mask;

    /* Walk until we find our key (already present) or an empty slot
       (fresh key). The loop always terminates because the table is
       never more than half full. */
    while (m->slots[i].used && m->slots[i].key != key)
        i = (i + 1) & m->mask;   /* '& mask' wraps to index 0 at the end */

    if (!m->slots[i].used) {
        m->slots[i].key = key;
        m->slots[i].used = 1;
        m->slots[i].val = 0.0;
    }
    m->slots[i].val += value;
}

/* Look up a key; returns 0.0 if the cell is not stored.
   Cells absent from the map are structurally zero, so 0.0 is exactly
   the right answer for a sparse matrix. */
static double cell_map_get(const CellMap *m, uint64_t key)
{
    size_t i = (size_t)mix64(key) & m->mask;

    while (m->slots[i].used) {
        if (m->slots[i].key == key) return m->slots[i].val;
        i = (i + 1) & m->mask;
    }
    return 0.0;
}

/* Fold a whole result matrix into a fresh CellMap.
   Returns 1 on success, 0 if memory ran out. */
static int build_cell_map(const TSparseMatrix *mat, CellMap *map)
{
    unsigned int i;

    if (!cell_map_init(map, mat->nnz ? (size_t)mat->nnz : 1)) return 0;

    for (i = 0; i < mat->nnz; ++i) {
        cell_map_add(map,
                     cell_key((unsigned int)mat->elements[i].r,
                              (unsigned int)mat->elements[i].c),
                     mat->elements[i].val);
    }
    return 1;
}

static void cell_map_free(CellMap *m)
{
    free(m->slots);
    m->slots = NULL;
    m->mask  = 0;
}

/* Check every cell present in map 'a' against map 'b'. Because 'b'
   returns 0.0 for absent cells, cells that exist on one side only
   are caught here too. Returns the number of mismatching cells. */
static size_t compare_direction(const CellMap *a, const CellMap *b,
                                const char *a_name, const char *b_name)
{
    size_t i;
    size_t mismatches = 0;

    for (i = 0; i <= a->mask; ++i) {
        double va, vb;

        if (!a->slots[i].used) continue;   /* empty slot, skip */

        va = a->slots[i].val;
        vb = cell_map_get(b, a->slots[i].key);

        /* fabs() alone is wrong here: NaN never compares greater than
           anything, so a NaN on either side would silently "pass".
           The explicit isnan() check closes that loophole. */
        if (isnan(va) || isnan(vb) || fabs(va - vb) > EPSILON) {
            unsigned int r = (unsigned int)(a->slots[i].key >> 32);
            unsigned int c = (unsigned int)(a->slots[i].key & 0xFFFFFFFFu);
            ++mismatches;
            if (mismatches <= MAX_REPORTED_MISMATCHES) {
                fprintf(stderr,
                        "  MISMATCH at cell (%u, %u): %s = %.17g, %s = %.17g\n",
                        r, c, a_name, va, b_name, vb);
            }
        }
    }
    return mismatches;
}

/* Full cell-by-cell comparison of the two results.
   Both directions are needed: forward catches cells that the parallel
   run got wrong; backward catches phantom cells that only exist in
   the parallel output. Tolerant to storage order and duplicates,
   strict about values and about cells existing on only one side. */
static bool verify_results(const TSparseMatrix *seq, const TSparseMatrix *par)
{
    CellMap seq_map, par_map;
    size_t forward, backward;
    bool ok;

    if (!seq || !par) {
        fprintf(stderr, "ERROR: a result matrix is NULL (engine failed).\n");
        return false;
    }

    if (seq->nrows != par->nrows || seq->ncols != par->ncols) {
        fprintf(stderr,
                "ERROR: dimension mismatch! sequential %u x %u vs parallel %u x %u\n",
                seq->nrows, seq->ncols, par->nrows, par->ncols);
        return false;
    }

    if (!build_cell_map(seq, &seq_map) || !build_cell_map(par, &par_map)) {
        fprintf(stderr, "ERROR: out of memory building comparison maps.\n");
        return false;
    }

    forward  = compare_direction(&seq_map, &par_map, "sequential", "parallel");
    backward = compare_direction(&par_map, &seq_map, "parallel", "sequential");

    if (forward + backward > MAX_REPORTED_MISMATCHES) {
        fprintf(stderr, "  ... (%zu additional mismatching cells)\n",
                forward + backward - MAX_REPORTED_MISMATCHES);
    }

    ok = (forward == 0) && (backward == 0);
    if (ok) {
        printf("  Cell-by-cell verification PASSED (tolerance %e, %u non-zeros).\n",
               EPSILON, seq->nnz);
    } else {
        fprintf(stderr, "  Verification FAILED: %zu forward + %zu backward mismatches.\n",
                forward, backward);
    }

    cell_map_free(&seq_map);
    cell_map_free(&par_map);
    return ok;
}

/* --------------------------------------------------------------- */
/* Driver                                                           */
/* --------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    const char *file_a, *file_b;
    int repeats = 3;
    int cfg, rep;
    double seq_time;
    TSparseMatrix *a, *b, *res_seq;
    double best_time[NUM_THREAD_CONFIGS];
    double avg_time[NUM_THREAD_CONFIGS];
    double speedup[NUM_THREAD_CONFIGS];
    bool passed[NUM_THREAD_CONFIGS];
    bool all_passed = true;

    if (argc < 3 || argc > 4) {
        fprintf(stderr,
                "Usage: %s <MatrixA.mtx> <MatrixB.mtx> [repeats]\n"
                "  repeats = runs per thread count in {1,2,4,8,16} (default 3)\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    file_a = argv[1];
    file_b = argv[2];
    if (argc == 4) {
        repeats = atoi(argv[3]);
        if (repeats < 1) repeats = 1;
        if (repeats > 100) repeats = 100;
    }

    printf("Loading sparse datasets into memory structures...\n");
    a = read_coo_matrix(file_a);
    b = read_coo_matrix(file_b);
    if (!a || !b) {
        fprintf(stderr, "ERROR: failed to load '%s' or '%s'.\n", file_a, file_b);
        free_sparse_matrix(a);
        free_sparse_matrix(b);
        return EXIT_FAILURE;
    }

    printf("Matrix A: %s -> %u x %u (%u non-zeros)\n",
           file_a, a->nrows, a->ncols, a->nnz);
    printf("Matrix B: %s -> %u x %u (%u non-zeros)\n",
           file_b, b->nrows, b->ncols, b->nnz);

    /* --- Step 1: sequential baseline. Run once; it is the reference
       all parallel runs are compared against, and its time is the
       numerator of every speedup figure. --- */
    printf("\n[1/2] Executing single-threaded baseline (ground truth)...\n");
    seq_time = now_seconds();
    res_seq = spmm_sequential(a, b);
    seq_time = now_seconds() - seq_time;

    if (!res_seq) {
        fprintf(stderr, "ERROR: sequential engine returned NULL.\n");
        free_sparse_matrix(a);
        free_sparse_matrix(b);
        return EXIT_FAILURE;
    }
    printf("  Sequential done in %.6f s (%u non-zeros in result)\n",
           seq_time, res_seq->nnz);

    /* --- Step 2: stress sweep. For each thread count, run the
       parallel engine 'repeats' times and verify every result. --- */
    printf("\n[2/2] Stress-testing parallel engine across thread counts "
           "(%d run(s) per configuration)...\n", repeats);

    for (cfg = 0; cfg < NUM_THREAD_CONFIGS; ++cfg) {
        int threads = THREAD_COUNTS[cfg];
        double total_time = 0.0;

        best_time[cfg] = HUGE_VAL;
        passed[cfg] = true;

        for (rep = 1; rep <= repeats; ++rep) {
            TSparseMatrix *res_par;
            double t;

            printf("  threads=%-2d run %d/%d ... ", threads, rep, repeats);
            fflush(stdout);

            t = now_seconds();
            res_par = spmm_parallel(a, b, threads);
            t = now_seconds() - t;

            if (!res_par) {
                fprintf(stderr, "\n    ERROR: parallel engine returned NULL.\n");
                passed[cfg] = false;
                break;
            }

            total_time += t;
            if (t < best_time[cfg]) best_time[cfg] = t;

            printf("(%.6f s) ", t);
            if (!verify_results(res_seq, res_par)) {
                passed[cfg] = false;
            }
            free_sparse_matrix(res_par);

            /* One failure proves the config is broken for this input;
               no need to hammer it further. */
            if (!passed[cfg]) break;
        }

        if (passed[cfg] && repeats > 0) {
            avg_time[cfg]  = total_time / (double)repeats;
            /* Speedup uses the BEST parallel time: slow runs are
               attributed to scheduling noise, not the algorithm. */
            speedup[cfg]   = seq_time / best_time[cfg];
        } else {
            avg_time[cfg]  = 0.0;
            speedup[cfg]   = 0.0;
        }
        all_passed = all_passed && passed[cfg];
    }

    /* --- Human-readable summary --- */
    printf("\n================ SUMMARY ================\n");
    printf("threads   best(s)    avg(s)    speedup   verdict\n");
    for (cfg = 0; cfg < NUM_THREAD_CONFIGS; ++cfg) {
        printf("%7d  %8.6f  %8.6f  %8.2fx   %s\n",
               THREAD_COUNTS[cfg],
               best_time[cfg] == HUGE_VAL ? 0.0 : best_time[cfg],
               avg_time[cfg],
               speedup[cfg],
               passed[cfg] ? "PASS" : "FAIL");
    }

    /* --- Machine-readable CSV for the report (tables + charts) --- */
    printf("\n--- CSV (for report data tables / speedup charts) ---\n");
    printf("matrix_a,matrix_b,threads,best_wall_time_s,avg_wall_time_s,speedup,status\n");
    for (cfg = 0; cfg < NUM_THREAD_CONFIGS; ++cfg) {
        printf("%s,%s,%d,%.6f,%.6f,%.3f,%s\n",
               file_a, file_b, THREAD_COUNTS[cfg],
               best_time[cfg] == HUGE_VAL ? 0.0 : best_time[cfg],
               avg_time[cfg],
               speedup[cfg],
               passed[cfg] ? "pass" : "fail");
    }

    free_sparse_matrix(res_seq);
    free_sparse_matrix(a);
    free_sparse_matrix(b);

    if (all_passed) {
        printf("\n>> SUCCESS: Parallel outputs match the sequential ground truth "
               "within EPSILON = %e for every thread count.\n", EPSILON);
        return EXIT_SUCCESS;
    }
    printf("\n>> FAILURE: Divergence detected between sequential and parallel engines.\n");
    return EXIT_FAILURE;
}
