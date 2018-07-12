/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.tokensencoder.charactersattention.CharsAttentionEncoderModel
import com.kotlinnlp.tokensencoder.charactersattention.CharsAttentionEncoderOptimizer
import com.kotlinnlp.tokensencoder.charactersbirnn.CharsBiRNNEncoderModel
import com.kotlinnlp.tokensencoder.charactersbirnn.CharsBiRNNEncoderOptimizer
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderOptimizer
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderOptimizer
import com.kotlinnlp.tokensencoder.morpho.MorphoEncoderModel
import com.kotlinnlp.tokensencoder.morpho.MorphoEncoderOptimizer

/**
 * The factory of [TokensEncoderOptimizer]s.
 */
object TokensEncoderOptimizerFactory {

  /**
   * @param model the model of a [TokensEncoder]
   *
   * @return a new instance of a [TokensEncoderOptimizer]
   */
  operator fun invoke(model: TokensEncoderModel, updateMethod: UpdateMethod<*>): TokensEncoderOptimizer =

    when (model) {
      is CharsAttentionEncoderModel -> CharsAttentionEncoderOptimizer(model, updateMethod)
      is CharsBiRNNEncoderModel -> CharsBiRNNEncoderOptimizer(model, updateMethod)
      is EmbeddingsEncoderModel -> EmbeddingsEncoderOptimizer(model, updateMethod)
      is MorphoEncoderModel -> MorphoEncoderOptimizer(model, updateMethod)
      is EnsembleTokensEncoderModel -> EnsembleTokensEncoderOptimizer(model, updateMethod)
      else -> throw RuntimeException("Invalid TokensEncoder model.")
    }
}