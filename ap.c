/*    
 *    File: ap.c
 *  Author: Wenfeng Yang<sysuyangwf@gmail.com>
 *          School of Software, Sun Yat-sen University<www.sysu.edu.cn>
 *    Date: 2012.02.22
 *   Title: Averaged Perceptron Algorithm for Wapiti<http://wapiti.limsi.fr/>
 *    How to use averaged perceptron algorithm (this ap.c) with Wapiti?
 *    [1] wapiti/src/wapiti.c:
 *        >       {"auto",   trn_auto }
 *        --- 
 *        <       {"auto",   trn_auto },
 *        <       {"ap",     trn_ap   }
 *    [2] wapiti/src/trainers.h
 *        < void trn_ap(mdl_t *mdl);
 *  Usage: 
 *       $(wapiti-path)/wapiti train --type crf --algo ap --maxiter 50 --devel \
 *       test.dat train.dat model
 */


#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "wapiti.h"
#include "decoder.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "tools.h"
#include "thread.h"
#include "vmath.h"
/******************************************************************************
 * Averaged Perceptron Algorithm
 *
 *   This section implement the averaged perceptron trainer. We use the averaged
 *   perceptron algorithm described by Collins[1].
 *   [1] Michael Collins. Discriminative training methods for hidden Markov
 *       models: theory and experiments with perceptron algorithms. Proceedings
 *       of the Conference on Empirical Methods in Natural Language Processing 
 *       (EMNLP 2002). 1-8. 2002.
 ******************************************************************************/

static uint32_t diff(uint32_t *x, uint32_t *y, uint32_t n) {
    uint32_t i, d = 0;
    for (i = 0; i < n; ++i) 
        if (x[i] != y[i]) 
            ++d;
    return d;
}

void trn_ap(mdl_t *mdl) {
	const uint64_t K  = mdl->nftr;
	const uint32_t I  = mdl->opt->maxiter;
	const uint32_t N  = mdl->train->nseq;
	double * x = mdl->theta;
	memset(x, 0.0, sizeof(double) * K);

	for (uint32_t i = 0; !uit_stop && i < I; ++i) {
		double loss = 0.0;
		for (uint32_t n = 0; n < N; ++n) {
			// Tag the sequence with the viterbi
			const seq_t *seq = mdl->train->seq[n];
			const uint32_t T = seq->len;
			uint32_t out[T], y[T];

			for (uint32_t t = 0; t < T; ++t)
				y[t] = seq->pos[t].lbl;

			//double  *psc = xmalloc(sizeof(double) * T);
			//double  *scs = xmalloc(sizeof(double)); 
			//tag_viterbi(mdl, seq, (uint32_t*)out, scs, (double *)psc);
			tag_viterbi(mdl, seq, (uint32_t*)out, NULL, NULL);

			uint32_t d = diff(out, y, T);
			if (d == 0)
				continue;

			for (uint32_t t = 0; t < T; ++t) {
				const pos_t *pos = &(seq->pos[t]);
				for (uint32_t j = 0; j < pos->ucnt; ++j) {
					uint64_t k = mdl->uoff[pos->uobs[j]];
					x[k + y[t]]   += 0.5;
					x[k + out[t]] -= 0.5;
				}
				for (uint32_t j = 0; j < pos->bcnt; ++j) {
					uint64_t k = mdl->boff[pos->bobs[j]];
					x[k + y[t]]   += 0.5;
					x[k + out[t]] -= 0.5;
				}
			}
			//free(psc);
			//free(scs);
			loss += d / (double) T;
		}

		if (uit_progress(mdl, i+1, loss) == false) // NA
			break;
	}
}
