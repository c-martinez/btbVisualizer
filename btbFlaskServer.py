from __future__ import division
from flask import Flask
from flask import jsonify
from flask import request
from flask import url_for
from flask import redirect

from operator import itemgetter
from collections import defaultdict

import numpy as np
import btb.utils.tools as btbtools
import btb.utils.wikiquery as wq

import mwclient
import pickle as pkl
import pycountry as pyc

import scipy.cluster.hierarchy

# Maybe wrap everything in a class ?
app = Flask(__name__)

wiki = mwclient.Site('en.wikipedia.org')
bots = wq.getAllBots(wiki)

sandersFeatures = pkl.load(open('sandersFeatures.pkl', 'r'))
sanders_featArray = sandersFeatures['features']
sanders_bookNames = np.array(sandersFeatures['titles'])
sanders_allCatNames = np.array(sandersFeatures['catNames'])

# Limit to N first books
# N = 10
# sanders_featArray = sanders_featArray[:N, :]
# sanders_bookNames = sanders_bookNames[:N]


def makeKeyMap(edits):
    keyMap = [{'country': k, 'score': v, 'countryCode': countryNumber(k)}
              for k, v in edits.iteritems()]
    return sorted(keyMap, key=itemgetter('score'), reverse=True)


def countryNumber(ctrAlpha):
    if ctrAlpha == 'UK':
        ctrAlpha = 'GB'
    return pyc.countries.get(alpha2=ctrAlpha).numeric


@app.route('/')
def index():
    return redirect(url_for('static', filename='clusters2.html'))


@app.route('/wikicontrib/<word>')
def interestInWord(word):

    ips, usrs, nrevs = wq.getContributionsForPage(wiki, word)

    if len(ips) == 0 and len(usrs) == 0 and nrevs == 0:
        return 'No wikipedia page found for: ', word

    knwRevs, conf, _, _, _, _ = btbtools.prepareData(ips, usrs, bots)
    knwRevsTotal = sum(knwRevs.values())
    knwRevs = {k: v/knwRevsTotal for k, v in knwRevs.iteritems()}
    expEdits = wq.getTotalContributions()

    # Keep only countries with known expected edits -- add 0 for missing ones
    knwRevs = defaultdict(lambda: 0, knwRevs)
    knwRevs = {k: knwRevs[k] for k, v in expEdits.iteritems()}

    # Relative score
    cmpEdits = btbtools.compareEdits(expEdits, knwRevs)
    cmpEdits = {k: v[2] for k, v in cmpEdits.iteritems()}

    # Format for Vega
    expEdits = makeKeyMap(expEdits)
    knwRevs = makeKeyMap(knwRevs)
    cmpEdits = makeKeyMap(cmpEdits)
    return jsonify(confidence=conf,
                   expectedScore=expEdits,
                   observedScore=knwRevs,
                   relativeScore=cmpEdits)


@app.route('/distances/')
def corpusDistances():
    if 'features[]' in request.args:
        featIdx = request.args.getlist('features[]')
        featIdx = [int(i) for i in featIdx]
    else:
        featIdx = range(len(sanders_allCatNames))

    bookNames = sanders_bookNames
    featArray = sanders_featArray[:, featIdx]
    allCatNames = sanders_allCatNames[featIdx]

    d = np.zeros((featArray.shape[0], featArray.shape[0]))
    for i in range(featArray.shape[0]):
        for j in range(featArray.shape[0]):
            d[i, j] = np.linalg.norm(featArray[i, :]-featArray[j, :])

    return jsonify(
                   bookNames=bookNames.tolist(),
                   featureNames=allCatNames.tolist(),
                   distances=d.tolist()
                   )


def newNode(node, maxDepth=4):
    treeNode = {}

    if (node.left or node.right) and maxDepth > 0:
        treeNode['children'] = []
        if node.left:
            kid = newNode(node.left, maxDepth=maxDepth-1)
            treeNode['children'].append(kid)
        if node.right:
            kid = newNode(node.right, maxDepth=maxDepth-1)
            treeNode['children'].append(kid)

    if node.is_leaf():
        treeNode['name'] = getName(node.id)
    elif maxDepth == 0:
        treeNode['name'] = 'Cluster containing %d books' % node.get_count()
    else:
        treeNode['name'] = ''
    return treeNode


def getName(bookId):
    return sanders_bookNames[bookId]


@app.route('/clusters/')
def clusters():
    if 'features[]' in request.args:
        featIdx = request.args.getlist('features[]')
        featIdx = [int(i) for i in featIdx]
    else:
        featIdx = range(len(sanders_allCatNames))

    if 'maxdepth' in request.args:
        maxDepth = int(request.args.get('maxdepth'))
    else:
        maxDepth = np.inf

    featArray = sanders_featArray[:, featIdx]

    d = np.zeros((featArray.shape[0], featArray.shape[0]))
    for i in range(featArray.shape[0]):
        for j in range(featArray.shape[0]):
            d[i, j] = np.linalg.norm(featArray[i, :]-featArray[j, :])

    Z = scipy.cluster.hierarchy.complete(d)
    tree = scipy.cluster.hierarchy.to_tree(Z)
    d3Dendro = newNode(tree, maxDepth=maxDepth)

    return jsonify(
                   d3Dendro,
                   )


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
