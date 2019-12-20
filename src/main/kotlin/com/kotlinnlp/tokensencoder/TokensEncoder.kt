/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * Encoder that generates a dense representation of sentence tokens.
 */
abstract class TokensEncoder<TokenType: Token, SentenceType: Sentence<TokenType>> :
  ItemsPool.IDItem,
  NeuralProcessor<SentenceType, List<DenseNDArray>, List<DenseNDArray>, NeuralProcessor.NoInputErrors> {

  /**
   * The tokens encoder model-
   */
  abstract val model: TokensEncoderModel<TokenType, SentenceType>

  /**
   * Do not propagate input errors during the backward.
   */
  override val propagateToInput: Boolean = false
}
