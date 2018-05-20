/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.affine

import com.kotlinnlp.tokensencoder.TokensEncoderBuilder

/**
 * A simple [AffineTokensEncoder] builder.
 *
 * @param model the encoder model
 * @param trainingMode whether the encoder is being trained
 */
class AffineTokensEncoderBuilder(model: AffineTokensEncoderModel, trainingMode: Boolean) : TokensEncoderBuilder {

  /**
   * The embeddings encoder.
   */
  private val affineEncoder = AffineTokensEncoder(model, trainingMode)

  /**
   * @return the [affineEncoder]
   */
  override operator fun invoke(): AffineTokensEncoder = this.affineEncoder
}
