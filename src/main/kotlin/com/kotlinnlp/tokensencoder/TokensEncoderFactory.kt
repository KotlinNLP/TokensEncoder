/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.SentenceIdentificable
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.linguisticdescription.sentence.token.TokenIdentificable
import com.kotlinnlp.tokensencoder.charactersattention.CharsAttentionEncoder
import com.kotlinnlp.tokensencoder.charactersattention.CharsAttentionEncoderModel
import com.kotlinnlp.tokensencoder.charactersbirnn.CharsBiRNNEncoder
import com.kotlinnlp.tokensencoder.charactersbirnn.CharsBiRNNEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoder
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoder
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import com.kotlinnlp.tokensencoder.lss.LSSTokensEncoder
import com.kotlinnlp.tokensencoder.lss.LSSTokensEncoderModel
import com.kotlinnlp.tokensencoder.morpho.MorphoEncoder
import com.kotlinnlp.tokensencoder.morpho.MorphoEncoderModel

/**
 * The factory of [TokensEncoder]s.
 */
object TokensEncoderFactory {

  /**
   * @param model the model of a [TokensEncoder]
   *
   * @return a new instance of a [TokensEncoder]
   */
  @Suppress("UNCHECKED_CAST")
  operator fun <TokenType : Token, SentenceType : Sentence<TokenType>>invoke(
    model: TokensEncoderModel<TokenType, SentenceType>,
    useDropout: Boolean,
    id: Int = 0
  ): TokensEncoder<TokenType, SentenceType> = when (model) {

    is CharsAttentionEncoderModel -> CharsAttentionEncoder(model = model, useDropout = useDropout, id = id)

    is CharsBiRNNEncoderModel -> CharsBiRNNEncoder(model = model, useDropout = useDropout, id = id)

    is EmbeddingsEncoderModel<TokenType, SentenceType> ->
      EmbeddingsEncoder(model = model, useDropout = useDropout, id = id)

    is MorphoEncoderModel -> MorphoEncoder(model = model, useDropout = useDropout, id = id)

    is EnsembleTokensEncoderModel -> EnsembleTokensEncoder(model = model, useDropout = useDropout, id = id)

    is LSSTokensEncoderModel -> LSSTokensEncoder(
      model = model as LSSTokensEncoderModel<TokenIdentificable, SentenceIdentificable<TokenIdentificable>>,
      useDropout = useDropout,
      id = id)

    else -> throw RuntimeException("Invalid TokensEncoder model ${model.javaClass.name}.")

  } as TokensEncoder<TokenType, SentenceType>
}
