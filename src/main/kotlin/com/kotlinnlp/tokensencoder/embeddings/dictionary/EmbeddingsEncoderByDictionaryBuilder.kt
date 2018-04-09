/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings.dictionary

import com.kotlinnlp.tokensencoder.TokensEncoderBuilder

/**
 * A simple [EmbeddingsEncoderByDictionaryBuilder] builder.
 *
 * @param model the encoder model
 * @param trainingMode whether the encoder is being trained
 */
class EmbeddingsEncoderByDictionaryBuilder(
  model: EmbeddingsEncoderByDictionaryModel,
  trainingMode: Boolean
) : TokensEncoderBuilder {

  /**
   * The embeddings encoder.
   */
  private val embeddingsEncoder = EmbeddingsEncoderByDictionary(model, trainingMode)

  /**
   * @return the [embeddingsEncoder]
   */
  override operator fun invoke(): EmbeddingsEncoderByDictionary = this.embeddingsEncoder
}
