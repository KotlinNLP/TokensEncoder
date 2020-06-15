/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [TokensEncoder]s.
 *
 * @param model the model of the tokens encoder
 */
class TokensEncodersPool<TokenType: Token, SentenceType: Sentence<TokenType>>(
  private val model: TokensEncoderModel<TokenType, SentenceType>
) : ItemsPool<TokensEncoder<TokenType, SentenceType>>() {

  /**
   * The factory of a new item.
   *
   * @param id the unique id of the item to create
   *
   * @return a new item with the given [id]
   */
  override fun itemFactory(id: Int): TokensEncoder<TokenType, SentenceType> =
    this.model.buildEncoder(id)

  /**
   * Release all the items of the pool and return a given number of available encoders.
   *
   * @param size the number of tokens encoder to return
   *
   * @return a list of tokens encoders
   */
  fun getEncoders(size: Int): List<TokensEncoder<TokenType, SentenceType>> {

    this.releaseAll()

    return List(size = size, init = { this.getItem() })
  }
}
