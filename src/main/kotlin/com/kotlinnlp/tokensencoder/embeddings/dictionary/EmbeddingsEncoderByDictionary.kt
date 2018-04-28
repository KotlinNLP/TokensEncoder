/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings.dictionary

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.simplednn.embeddings.Embedding
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoder

/**
 * The tokens encoder that encodes a token using an embeddings map.
 *
 * @property model the model of this tokens encoder
 * @property trainingMode whether the encoder is being trained
 */
class EmbeddingsEncoderByDictionary(
  private val model: EmbeddingsEncoderByDictionaryModel,
  private val trainingMode: Boolean
) : EmbeddingsEncoder(
  model = model,
  trainingMode = trainingMode) {

  /**
   * @param token a token
   *
   * @return the embedding resulting from the given [token]
   */
  override fun getEmbedding(token: Token): Embedding =
    this.model.embeddingsMap.get(
      element = this.model.tokenEmbeddingKey(token),
      dropoutCoefficient = if (this.trainingMode) this.model.dropoutCoefficient else 0.0
    )
}