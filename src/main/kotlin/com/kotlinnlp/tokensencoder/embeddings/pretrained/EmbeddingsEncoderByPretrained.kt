/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings.pretrained

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoder

/**
 * The tokens encoder that encodes a token using a mao of pre-trained embeddings..
 *
 * @property model the model of this tokens encoder
 * @property trainingMode whether the encoder is being trained
 */
class EmbeddingsEncoderByPretrained(
  private val model: EmbeddingsEncoderByPretrainedModel,
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
    this.model.embeddingsMap.get(this.model.tokenEmbeddingKey(token))
}