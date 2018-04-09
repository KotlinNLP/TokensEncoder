/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charactersattention

import com.kotlinnlp.tokensencoder.TokensEncoderBuilder

/**
 * A simple [CharsAttentionEncoder] builder.
 *
 * @param model the encoder model
 */
class CharsAttentionEncoderBuilder(model: CharsAttentionEncoderModel, trainingMode: Boolean) : TokensEncoderBuilder {

  /**
   * The characters encoder.
   */
  private val charactersEncoder = CharsAttentionEncoder(model, trainingMode)

  /**
   * @return the [charactersEncoder]
   */
  override operator fun invoke(): CharsAttentionEncoder = this.charactersEncoder
}
