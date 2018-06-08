/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Encoder that generates a dense representation of the sentence tokens.
 */
interface TokensEncoder {

  /**
   * Encode a list of tokens.
   *
   * @param tokens a list of [Token]
   *
   * @return a list of the same size of the [tokens] with their encoded representation
   */
  fun encode(tokens: List<Token>): List<DenseNDArray>

  /**
   * Propagate the errors.
   *
   * @param errors the errors of the current encoding
   */
  fun backward(errors: List<DenseNDArray>)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [TokensEncoderParameters] parameters
   */
  fun getParamsErrors(copy: Boolean = true): TokensEncoderParameters
}
