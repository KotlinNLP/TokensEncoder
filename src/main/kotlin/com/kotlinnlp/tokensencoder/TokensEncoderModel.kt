/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import java.io.Serializable

/**
 * The model of a tokens-encoder.
 */
@Suppress("UNUSED")
interface TokensEncoderModel<TokenType: Token, SentenceType: Sentence<TokenType>> : Serializable {

  /**
   * The size of the token encoding vectors.
   */
  val tokenEncodingSize: Int

  /**
   * @param useDropout whether to apply the dropout
   * @param id an identification number useful to track a specific encoder
   *
   * @return a new tokens encoder that uses this model
   */
  fun buildEncoder(useDropout: Boolean, id: Int = 0): TokensEncoder<TokenType, SentenceType>

  /**
   * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
   *
   * @return a new optimizer for this model
   */
  fun buildOptimizer(updateMethod: UpdateMethod<*>): TokensEncoderOptimizer
}
