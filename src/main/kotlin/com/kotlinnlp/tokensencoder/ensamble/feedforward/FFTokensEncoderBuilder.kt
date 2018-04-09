/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.feedforward

import com.kotlinnlp.tokensencoder.TokensEncoderBuilder

/**
 * A simple [FFTokensEncoder] builder.
 *
 * @param model the encoder model
 * @param trainingMode whether the encoder is being trained
 */
class FFTokensEncoderBuilder(model: FFTokensEncoderModel, trainingMode: Boolean) : TokensEncoderBuilder {

  /**
   * The embeddings encoder.
   */
  private val ffEncoder = FFTokensEncoder(model, trainingMode)

  /**
   * @return the [ffEncoder]
   */
  override operator fun invoke(): FFTokensEncoder  = this.ffEncoder
}
