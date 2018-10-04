/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charlm

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.tokensencoder.TokensEncoderModel

/**
 * The model of the [CharLMEncoder].
 *
 * @param charLM
 * @param revCharLM
 */
class CharLMEncoderModel(
  val charLM: CharLM,
  val revCharLM: CharLM
) : TokensEncoderModel<FormToken, Sentence<FormToken>> {

  /**
   * The size of the token encoding vectors.
   */
  override val tokenEncodingSize = this.charLM.recurrentNetwork.outputSize + this.charLM.recurrentNetwork.outputSize

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  init {
    require(!this.charLM.reverseModel) { "The charLM must be trained to process the sequence from left to right."}
    require(this.revCharLM.reverseModel) { "The revCharLM must be trained to process the sequence from right to left."}
  }

  /**
   * @param useDropout whether to apply the dropout
   * @param id an identification number useful to track a specific encoder
   *
   * @return a new tokens encoder that uses this model
   */
  override fun buildEncoder(useDropout: Boolean, id: Int) = CharLMEncoder(model = this, id = id)

  /**
   * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
   *
   * @return a new optimizer for this model
   */
  override fun buildOptimizer(updateMethod: UpdateMethod<*>) = CharLMEncoderOptimizer(
    model = this,
    updateMethod = updateMethod
  )
}