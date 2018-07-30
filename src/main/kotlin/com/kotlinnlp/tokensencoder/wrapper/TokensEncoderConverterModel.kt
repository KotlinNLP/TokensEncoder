/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.wrapper

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.tokensencoder.TokensEncoderFactory
import com.kotlinnlp.tokensencoder.TokensEncoderModel

/**
 * A container of a [TokensEncoderModel] and the related [SentenceConverter] used to obtain the required kind of
 * sentence.
 *
 * @property model a tokens encoder model
 * @property converter the sentence converter to obtain the kind of sentence required by the [model]
 */
data class TokensEncoderConverterModel<
  FromTokenType: Token,
  FromSentenceType: Sentence<FromTokenType>,
  ToTokenType: Token,
  ToSentenceType: Sentence<ToTokenType>>
(
  val model: TokensEncoderModel<ToTokenType, ToSentenceType>,
  val converter: SentenceConverter<FromTokenType, FromSentenceType, ToTokenType, ToSentenceType>
) {

  /**
   * @param useDropout whether to apply the dropout
   *
   * @return a tokens encoder wrapper compatible with this [model] and [converter]
   */
  fun buildWrapper(useDropout: Boolean) = TokensEncoderWrapper(
    encoder = TokensEncoderFactory(this.model, useDropout = useDropout),
    converter = this.converter
  )
}
