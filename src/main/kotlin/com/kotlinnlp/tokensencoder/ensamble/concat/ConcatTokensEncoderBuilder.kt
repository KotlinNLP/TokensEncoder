/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.concat

import com.kotlinnlp.tokensencoder.TokensEncoderBuilder

/**
 * A simple [ConcatTokensEncoder] builder.
 *
 * @param model the encoder model
 * @param trainingMode whether the encoder is being trained
 */
class ConcatTokensEncoderBuilder(model: ConcatTokensEncoderModel, trainingMode: Boolean) : TokensEncoderBuilder {

  /**
   * The embeddings encoder.
   */
  private val concatEncoder = ConcatTokensEncoder(model, trainingMode)

  /**
   * @return the [concatEncoder]
   */
  override operator fun invoke(): ConcatTokensEncoder = this.concatEncoder
}
