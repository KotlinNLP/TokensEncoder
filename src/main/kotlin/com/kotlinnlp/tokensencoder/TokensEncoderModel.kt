/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder

/**
 * The model of a tokens-encoder.
 */
interface TokensEncoderModel {

  /**
   * The size of the token encoding vectors.
   */
  val tokenEncodingSize: Int
}
