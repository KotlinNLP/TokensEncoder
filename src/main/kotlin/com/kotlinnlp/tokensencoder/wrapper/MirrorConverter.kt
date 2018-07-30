/* Copyright 2017-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.wrapper

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token

/**
 * The sentence converter that mirrors the input sentence to the output.
 */
class MirrorConverter<TokenType: Token, SentenceType: Sentence<TokenType>>
  : SentenceConverter<TokenType, SentenceType, TokenType, SentenceType> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Return the same sentence given in input.
   *
   * @param sentence the input sentence
   *
   * @return the same sentence given in input
   */
  override fun convert(sentence: SentenceType): SentenceType = sentence
}
