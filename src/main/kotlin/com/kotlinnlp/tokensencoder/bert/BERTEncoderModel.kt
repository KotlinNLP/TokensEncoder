/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.bert

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.deeplearning.transformers.BERTModel
import com.kotlinnlp.tokensencoder.TokensEncoderModel

/**
 * The model of the [BERTEncoder].
 *
 * @property bert a BERT model
 */
class BERTEncoderModel<TokenType: FormToken, SentenceType: Sentence<TokenType>>(
  val bert: BERTModel
) : TokensEncoderModel<TokenType, SentenceType> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
  /**
   * The size of the token encoding vectors.
   */
  override val tokenEncodingSize: Int = this.bert.outputSize

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = """
    encoding size %d
  """.trimIndent().format(
    this.tokenEncodingSize
  )

  /**
   * @param id an identification number useful to track a specific encoder
   *
   * @return a new tokens encoder that uses this model
   */
  override fun buildEncoder(id: Int) = BERTEncoder(model = this, id = id)
}
