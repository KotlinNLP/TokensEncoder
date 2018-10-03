/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.lss

import com.kotlinnlp.linguisticdescription.sentence.SentenceIdentificable
import com.kotlinnlp.linguisticdescription.sentence.token.TokenIdentificable
import com.kotlinnlp.lssencoder.LSSModel
import com.kotlinnlp.tokensencoder.TokensEncoderModel

/**
 * The model of an [LSSTokensEncoder].
 *
 */
class LSSTokensEncoderModel<TokenType : TokenIdentificable, SentenceType : SentenceIdentificable<TokenType>>(
  internal val lssModel: LSSModel<TokenType, SentenceType>
) : TokensEncoderModel<TokenType, SentenceType> {

  /**
   * The size of the tokens encodings (context vectors + latent head representation).
   * Note: the context vectors size is equal to the latent head representations size.
   */
  override val tokenEncodingSize: Int = 2 * this.lssModel.contextVectorsSize
}
