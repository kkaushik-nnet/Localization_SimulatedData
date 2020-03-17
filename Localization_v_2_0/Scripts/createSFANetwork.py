#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Creates a four-layer hierarchical SFA network 
# Input Image dimension: 120 x 120
# Output dimension: 8
#
# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.
#
#

import mdp


def createSFANetwork():
    # ===========================================
    # Parameters
    # ===========================================
    IMAGE_SIZE = 120, 120
    IMAGE_DEPTH = 1
    PATCH_SIZE = 10, 10
    SPACING = 5, 5

    # ===========================================
    # input layer
    # ===========================================
    switchboard = mdp.hinet.Rectangular2dSwitchboard(in_channels_xy=IMAGE_SIZE,
                                                     field_channels_xy=PATCH_SIZE,
                                                     field_spacing_xy=SPACING,
                                                     in_channel_dim=IMAGE_DEPTH)

    sfa1 = mdp.nodes.SFANode(input_dim=switchboard.out_channel_dim, output_dim=20, include_last_sample=True)
    quadraticExpansion = mdp.nodes.QuadraticExpansionNode(input_dim=sfa1.output_dim)
    sfa2 = mdp.nodes.SFANode(input_dim=quadraticExpansion.output_dim, output_dim=12, include_last_sample=True)
    flowNode = mdp.hinet.FlowNode(sfa1 + quadraticExpansion + sfa2)
    inputLayer = mdp.hinet.CloneLayer(flowNode, n_nodes=switchboard.output_channels)

    # ===========================================
    # layer 1
    # ===========================================
    switchboard2 = mdp.hinet.Rectangular2dSwitchboard(in_channels_xy=switchboard.out_channels_xy,
                                                      field_channels_xy=(5, 5),
                                                      field_spacing_xy=(2, 2),
                                                      in_channel_dim=sfa2.output_dim)

    sfa1 = mdp.nodes.SFANode(input_dim=switchboard2.out_channel_dim, output_dim=20, include_last_sample=True)
    quadraticExpansion = mdp.nodes.QuadraticExpansionNode(input_dim=sfa1.output_dim)
    sfa2 = mdp.nodes.SFANode(input_dim=quadraticExpansion.output_dim, output_dim=12, include_last_sample=True)
    flowNode = mdp.hinet.FlowNode(sfa1 + quadraticExpansion + sfa2)
    middleLayer = mdp.hinet.CloneLayer(flowNode, n_nodes=switchboard2.output_channels)

    # ===========================================
    # layer 2
    # ===========================================
    switchboard3 = mdp.hinet.Rectangular2dSwitchboard(in_channels_xy=switchboard2.out_channels_xy,
                                                      field_channels_xy=(4, 4),
                                                      field_spacing_xy=(2, 2),
                                                      in_channel_dim=sfa2.output_dim)

    sfa1 = mdp.nodes.SFANode(input_dim=switchboard3.out_channel_dim, output_dim=20, include_last_sample=True)
    quadraticExpansion = mdp.nodes.QuadraticExpansionNode(input_dim=sfa1.output_dim)
    sfa2 = mdp.nodes.SFANode(input_dim=quadraticExpansion.output_dim, output_dim=12, include_last_sample=True)
    flowNode3 = mdp.hinet.FlowNode(sfa1 + quadraticExpansion + sfa2)
    middleLayer2 = mdp.hinet.CloneLayer(flowNode3, n_nodes=switchboard3.output_channels)

    # ===========================================
    # output layer
    # ===========================================
    sfa1 = mdp.nodes.SFANode(input_dim=middleLayer2.output_dim, output_dim=20, include_last_sample=True)
    quadraticExpansion = mdp.nodes.QuadraticExpansionNode(input_dim=sfa1.output_dim)
    sfa2 = mdp.nodes.SFANode(input_dim=quadraticExpansion.output_dim, output_dim=8, include_last_sample=True)
    outputLayer = mdp.hinet.FlowNode(sfa1 + quadraticExpansion + sfa2)

    sfaNet = mdp.Flow([switchboard, inputLayer, switchboard2, middleLayer, switchboard3, middleLayer2, outputLayer],
                      verbose=True)
    return sfaNet
