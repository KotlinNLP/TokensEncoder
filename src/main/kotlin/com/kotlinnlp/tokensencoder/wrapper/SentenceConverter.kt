/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.wrapper

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token

/**
 * The sentence converter associated to a [TokensEncoder] and used to dynamically convert an input sentence into the
 * sentence required by it.
 */
interface SentenceConverter<
  FromTokenType: Token,
  FromSentenceType: Sentence<FromTokenType>,
  ToTokenType: Token,
  ToSentenceType: Sentence<ToTokenType>
  > {

  /**
   * Convert a given sentence from a type to another.
   *
   * @param sentence the input sentence
   *
   * @return a new converted sentence of the output type
   */
  fun convert(sentence: FromSentenceType): ToSentenceType
}
