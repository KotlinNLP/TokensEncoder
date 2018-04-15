/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder

import com.kotlinnlp.tokensencoder.charactersattention.CharsAttentionEncoderBuilder
import com.kotlinnlp.tokensencoder.charactersattention.CharsAttentionEncoderModel
import com.kotlinnlp.tokensencoder.charactersbirnn.CharsBiRNNEncoderBuilder
import com.kotlinnlp.tokensencoder.charactersbirnn.CharsBiRNNEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.dictionary.EmbeddingsEncoderByDictionaryBuilder
import com.kotlinnlp.tokensencoder.embeddings.dictionary.EmbeddingsEncoderByDictionaryModel
import com.kotlinnlp.tokensencoder.embeddings.pretrained.EmbeddingsEncoderByPretrainedBuilder
import com.kotlinnlp.tokensencoder.embeddings.pretrained.EmbeddingsEncoderByPretrainedModel
import com.kotlinnlp.tokensencoder.ensamble.concat.ConcatTokensEncoderBuilder
import com.kotlinnlp.tokensencoder.ensamble.concat.ConcatTokensEncoderModel
import com.kotlinnlp.tokensencoder.ensamble.feedforward.FFTokensEncoderBuilder
import com.kotlinnlp.tokensencoder.ensamble.feedforward.FFTokensEncoderModel
import com.kotlinnlp.tokensencoder.morpho.MorphoEncoderBuilder
import com.kotlinnlp.tokensencoder.morpho.MorphoEncoderModel

/**
 * The factory of [TokensEncoder]s.
 */
object TokensEncoderFactory {

  /**
   * @param model the model of a [TokensEncoder]
   *
   * @return a new instance of a [TokensEncoderBuilder]
   */
  operator fun invoke(model: TokensEncoderModel, trainingMode: Boolean = false): TokensEncoderBuilder =
    when (model) {
      is CharsAttentionEncoderModel -> CharsAttentionEncoderBuilder(model, trainingMode)
      is CharsBiRNNEncoderModel -> CharsBiRNNEncoderBuilder(model, trainingMode)
      is EmbeddingsEncoderByDictionaryModel -> EmbeddingsEncoderByDictionaryBuilder(model, trainingMode)
      is EmbeddingsEncoderByPretrainedModel -> EmbeddingsEncoderByPretrainedBuilder(model, trainingMode)
      is FFTokensEncoderModel -> FFTokensEncoderBuilder(model, trainingMode)
      is ConcatTokensEncoderModel -> ConcatTokensEncoderBuilder(model, trainingMode)
      is MorphoEncoderModel -> MorphoEncoderBuilder(model, trainingMode)
      else -> throw RuntimeException("Invalid TokensEncoder model ${model::javaClass.name}.")
    }
}
