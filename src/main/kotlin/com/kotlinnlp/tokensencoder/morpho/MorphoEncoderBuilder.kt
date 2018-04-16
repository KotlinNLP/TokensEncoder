/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho

import com.kotlinnlp.tokensencoder.TokensEncoderBuilder

/**
 * A simple [MorphoEncoder] builder.
 *
 * @param model the encoder model
 * @param trainingMode whether the encoder is being trained
 */
class MorphoEncoderBuilder(
  model: MorphoEncoderModel,
  trainingMode: Boolean
) : TokensEncoderBuilder {

  /**
   * The characters encoder.
   */
  private val morphoEncoder = MorphoEncoder(model, trainingMode)

  /**
   * @return the [morphoEncoder]
   */
  override operator fun invoke(): MorphoEncoder = this.morphoEncoder
}
