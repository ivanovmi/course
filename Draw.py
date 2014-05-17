__author__ = 'michael'


def draw(G, pos=None, ax=None, hold=None, **kwds):
    try:
        import matplotlib.pylab as pylab
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    cf=pylab.gcf()
    cf.set_facecolor('w')
    if ax is None:
        if cf._axstack() is None:
            ax=cf.add_axes((0,0,1,1))
        else:
            ax=cf.gca()

 # allow callers to override the hold state by passing hold=True|False
    b = pylab.ishold()
    h = kwds.pop('hold', None)
    if h is not None:
        pylab.hold(h)
    try:
        draw_networkx(G,pos=pos,ax=ax,**kwds)
        ax.set_axis_off()
        pylab.draw_if_interactive()
    except:
        pylab.hold(b)
        raise
    pylab.hold(b)
    return

def draw_networkx(G, pos=None, with_labels=True, **kwds):
    try:
        import matplotlib.pylab as pylab
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if pos is None:
        pos=spring_layout(G) # default to spring layout

    node_collection=draw_networkx_nodes(G, pos, **kwds)
    edge_collection=draw_networkx_edges(G, pos, **kwds)
    if with_labels:
        draw_networkx_labels(G, pos, **kwds)
    pylab.draw_if_interactive()


def fruchterman_reingold_layout(G,dim=2, pos=None, fixed=None, iterations=50, weight='weight', scale=1):
    try:
        import numpy as np
    except ImportError:
        raise ImportError("fruchterman_reingold_layout() requires numpy: http://scipy.org/ ")
    if fixed is not None:
        nfixed=dict(zip(G,range(len(G))))
        fixed=np.asarray([nfixed[v] for v in fixed])

    if pos is not None:
        pos_arr=np.asarray(np.random.random((len(G),dim)))
        for i,n in enumerate(G):
            if n in pos:
                pos_arr[i]=np.asarray(pos[n])
    else:
        pos_arr=None

    if len(G)==0:
        return {}
    if len(G)==1:
        return {G.nodes()[0]:(1,)*dim}

    try:
        # Sparse matrix
        if len(G) < 500:  # sparse solver for large graphs
            raise ValueError
        A=to_scipy_sparse_matrix(G,weight=weight)
        pos=_sparse_fruchterman_reingold(A,dim,pos_arr,fixed,iterations)
    except:
        A=to_numpy_matrix(G,weight=weight)
        pos=_fruchterman_reingold(A,dim,pos_arr,fixed,iterations)
    if fixed is None:
        pos=_rescale_layout(pos,scale=scale)
    return dict(zip(G,pos))

spring_layout=fruchterman_reingold_layout

def _fruchterman_reingold(A, dim=2, pos=None, fixed=None, iterations=50):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    try:
        import numpy as np
    except ImportError:
        raise ImportError("_fruchterman_reingold() requires numpy: http://scipy.org/ ")

    try:
        nnodes,_=A.shape
    except AttributeError:
        raise NetworkXError(
            "fruchterman_reingold() takes an adjacency matrix as input")

    A=np.asarray(A) # make sure we have an array instead of a matrix

    if pos==None:
        # random initial positions
        pos=np.asarray(np.random.random((nnodes,dim)),dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos=pos.astype(A.dtype)

    # optimal distance between nodes
    k=np.sqrt(1.0/nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t=0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt=t/float(iterations+1)
    delta = np.zeros((pos.shape[0],pos.shape[0],pos.shape[1]),dtype=A.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # matrix of difference between points
        for i in range(pos.shape[1]):
            delta[:,:,i]= pos[:,i,None]-pos[:,i]
        # distance between points
        distance=np.sqrt((delta**2).sum(axis=-1))
        # enforce minimum distance of 0.01
        distance=np.where(distance<0.01,0.01,distance)
        # displacement "force"
        displacement=np.transpose(np.transpose(delta)*\
                                  (k*k/distance**2-A*distance/k))\
                                  .sum(axis=1)
        # update positions
        length=np.sqrt((displacement**2).sum(axis=1))
        length=np.where(length<0.01,0.1,length)
        delta_pos=np.transpose(np.transpose(displacement)*t/length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed]=0.0
        pos+=delta_pos
        # cool temperature
        t-=dt

    return pos


def _sparse_fruchterman_reingold(A, dim=2, pos=None, fixed=None, iterations=50):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    # Sparse version
    try:
        import numpy as np
    except ImportError:
        raise ImportError("_sparse_fruchterman_reingold() requires numpy: http://scipy.org/ ")
    try:
        nnodes,_=A.shape
    except AttributeError:
        raise NetworkXError(
            "fruchterman_reingold() takes an adjacency matrix as input")
    try:
        from scipy.sparse import spdiags,coo_matrix
    except ImportError:
        raise ImportError("_sparse_fruchterman_reingold() scipy numpy: http://scipy.org/ ")
    # make sure we have a LIst of Lists representation
    try:
        A=A.tolil()
    except:
        A=(coo_matrix(A)).tolil()

    if pos==None:
        # random initial positions
        pos=np.asarray(np.random.random((nnodes,dim)),dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos=pos.astype(A.dtype)

    # no fixed nodes
    if fixed==None:
        fixed=[]

    # optimal distance between nodes
    k=np.sqrt(1.0/nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t=0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt=t/float(iterations+1)

    displacement=np.zeros((dim,nnodes))
    for iteration in range(iterations):
        displacement*=0
        # loop over rows
        for i in range(A.shape[0]):
            if i in fixed:
                continue
            # difference between this row's node position and all others
            delta=(pos[i]-pos).T
            # distance between points
            distance=np.sqrt((delta**2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance=np.where(distance<0.01,0.01,distance)
            # the adjacency matrix row
            Ai=np.asarray(A.getrowview(i).toarray())
            # displacement "force"
            displacement[:,i]+=\
                (delta*(k*k/distance**2-Ai*distance/k)).sum(axis=1)
        # update positions
        length=np.sqrt((displacement**2).sum(axis=0))
        length=np.where(length<0.01,0.1,length)
        pos+=(displacement*t/length).T
        # cool temperature
        t-=dt

    return pos

def _rescale_layout(pos,scale=1):
    # rescale to (0,pscale) in all axes

    # shift origin to (0,0)
    lim=0 # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:,i]-=pos[:,i].min()
        lim=max(pos[:,i].max(),lim)
    # rescale to (0,scale) in all directions, preserves aspect
    for i in range(pos.shape[1]):
        pos[:,i]*=scale/lim
    return pos

def draw_networkx_nodes(G, pos,
                        nodelist=None,
                        node_size=300,
                        node_color='r',
                        node_shape='o',
                        alpha=1.0,
                        cmap=None,
                        vmin=None,
                        vmax=None,
                        ax=None,
                        linewidths=None,
                        label = None,
                        **kwds):
    try:
        import matplotlib.pylab as pylab
        import numpy
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise


    if ax is None:
        ax=pylab.gca()

    if nodelist is None:
        nodelist=G.nodes()

    if not nodelist or len(nodelist)==0:  # empty nodelist, no drawing
        return None

    try:
        xy=numpy.asarray([pos[v] for v in nodelist])
    except KeyError as e:
        raise NetworkXError('Node %s has no position.'%e)
    except ValueError:
        raise NetworkXError('Bad value in node positions.')



    node_collection=ax.scatter(xy[:,0], xy[:,1],
                               s=node_size,
                               c=node_color,
                               marker=node_shape,
                               cmap=cmap,
                               vmin=vmin,
                               vmax=vmax,
                               alpha=alpha,
                               linewidths=linewidths,
                               label=label)

#    pylab.axes(ax)
    pylab.sci(node_collection)
    node_collection.set_zorder(2)
    return node_collection


def draw_networkx_edges(G, pos,
                        edgelist=None,
                        width=1.0,
                        edge_color='k',
                        style='solid',
                        alpha=None,
                        edge_cmap=None,
                        edge_vmin=None,
                        edge_vmax=None,
                        ax=None,
                        arrows=True,
                        label=None,
                        **kwds):
    try:
        import matplotlib
        import matplotlib.pylab as pylab
        import matplotlib.cbook as cb
        from matplotlib.colors import colorConverter,Colormap
        from matplotlib.collections import LineCollection
        import numpy
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax=pylab.gca()

    if edgelist is None:
        edgelist=G.edges()

    if not edgelist or len(edgelist)==0: # no edges!
        return None

    # set edge positions
    edge_pos=numpy.asarray([(pos[e[0]],pos[e[1]]) for e in edgelist])

    if not cb.iterable(width):
        lw = (width,)
    else:
        lw = width

    if not cb.is_string_like(edge_color) \
           and cb.iterable(edge_color) \
           and len(edge_color)==len(edge_pos):
        if numpy.alltrue([cb.is_string_like(c)
                         for c in edge_color]):
            edge_colors = tuple([colorConverter.to_rgba(c,alpha)
                                 for c in edge_color])
        elif numpy.alltrue([not cb.is_string_like(c)
                           for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if numpy.alltrue([cb.iterable(c) and len(c) in (3,4)
                             for c in edge_color]):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must consist of either color names or numbers')
    else:
        if cb.is_string_like(edge_color) or len(edge_color)==1:
            edge_colors = ( colorConverter.to_rgba(edge_color, alpha), )
        else:
            raise ValueError('edge_color must be a single color or list of exactly m colors where m is the number or edges')

    edge_collection = LineCollection(edge_pos, colors=edge_colors, linewidths=lw, antialiaseds=(1,), linestyle=style, transOffset=ax.transData,)


    edge_collection.set_zorder(1) # edges go behind nodes
    edge_collection.set_label(label)
    ax.add_collection(edge_collection)
    if cb.is_numlike(alpha):
        edge_collection.set_alpha(alpha)

    if edge_colors is None:
        if edge_cmap is not None:
            assert(isinstance(edge_cmap, Colormap))
        edge_collection.set_array(numpy.asarray(edge_color))
        edge_collection.set_cmap(edge_cmap)
        if edge_vmin is not None or edge_vmax is not None:
            edge_collection.set_clim(edge_vmin, edge_vmax)
        else:
            edge_collection.autoscale()
        pylab.sci(edge_collection)

    arrow_collection=None

    if G.is_directed() and arrows:

        arrow_colors = edge_colors
        a_pos=[]
        p=1.0-0.25 # make head segment 25 percent of edge length
        for src,dst in edge_pos:
            x1,y1=src
            x2,y2=dst
            dx=x2-x1 # x offset
            dy=y2-y1 # y offset
            d=numpy.sqrt(float(dx**2+dy**2)) # length of edge
            if d==0: # source and target at same position
                continue
            if dx==0: # vertical edge
                xa=x2
                ya=dy*p+y1
            if dy==0: # horizontal edge
                ya=y2
                xa=dx*p+x1
            else:
                theta=numpy.arctan2(dy,dx)
                xa=p*d*numpy.cos(theta)+x1
                ya=p*d*numpy.sin(theta)+y1

            a_pos.append(((xa,ya),(x2,y2)))

        arrow_collection = LineCollection(a_pos,
                                colors       = arrow_colors,
                                linewidths   = [4*ww for ww in lw],
                                antialiaseds = (1,),
                                transOffset = ax.transData,
                                )

        arrow_collection.set_zorder(1) # edges go behind nodes
        arrow_collection.set_label(label)
        ax.add_collection(arrow_collection)


    # update view
    minx = numpy.amin(numpy.ravel(edge_pos[:,:,0]))
    maxx = numpy.amax(numpy.ravel(edge_pos[:,:,0]))
    miny = numpy.amin(numpy.ravel(edge_pos[:,:,1]))
    maxy = numpy.amax(numpy.ravel(edge_pos[:,:,1]))

    w = maxx-minx
    h = maxy-miny
    padx, pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    ax.update_datalim( corners)
    ax.autoscale_view()

#    if arrow_collection:

    return edge_collection


def draw_networkx_labels(G, pos,
                         labels=None,
                         font_size=12,
                         font_color='k',
                         font_family='sans-serif',
                         font_weight='normal',
                         alpha=1.0,
                         ax=None,
                         **kwds):
    try:
        import matplotlib.pylab as pylab
        import matplotlib.cbook as cb
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax=pylab.gca()

    if labels is None:
        labels=dict( (n,n) for n in G.nodes())

    # set optional alignment
    horizontalalignment=kwds.get('horizontalalignment','center')
    verticalalignment=kwds.get('verticalalignment','center')

    text_items={}  # there is no text collection so we'll fake one
    for n, label in labels.items():
        (x,y)=pos[n]
        if not cb.is_string_like(label):
            label=str(label) # this will cause "1" and 1 to be labeled the same
        t=ax.text(x, y,
                  label,
                  size=font_size,
                  color=font_color,
                  family=font_family,
                  weight=font_weight,
                  horizontalalignment=horizontalalignment,
                  verticalalignment=verticalalignment,
                  transform = ax.transData,
                  clip_on=True,
                  )
        text_items[n]=t

    return text_items

def to_numpy_matrix(G, nodelist=None, dtype=None, order=None,
                    multigraph_weight=sum, weight='weight'):
    try:
        import numpy as np
    except ImportError:
        raise ImportError(\
          "to_numpy_matrix() requires numpy: http://scipy.org/ ")

    if nodelist is None:
        nodelist = G.nodes()

    nodeset = set(nodelist)
    if len(nodelist) != len(nodeset):
        msg = "Ambiguous ordering: `nodelist` contained duplicates."
        raise NetworkXError(msg)

    nlen=len(nodelist)
    undirected = not G.is_directed()
    index=dict(zip(nodelist,range(nlen)))

    if G.is_multigraph():
        # Handle MultiGraphs and MultiDiGraphs
        # array of nan' to start with, any leftover nans will be converted to 0
        # nans are used so we can use sum, min, max for multigraphs
        M = np.zeros((nlen,nlen), dtype=dtype, order=order)+np.nan
        # use numpy nan-aware operations
        operator={sum:np.nansum, min:np.nanmin, max:np.nanmax}
        try:
            op=operator[multigraph_weight]
        except:
            raise ValueError('multigraph_weight must be sum, min, or max')

        for u,v,attrs in G.edges_iter(data=True):
            if (u in nodeset) and (v in nodeset):
                i,j = index[u],index[v]
                e_weight = attrs.get(weight, 1)
                M[i,j] = op([e_weight,M[i,j]])
                if undirected:
                    M[j,i] = M[i,j]
        # convert any nans to zeros
        M = np.asmatrix(np.nan_to_num(M))
    else:
        # Graph or DiGraph, this is much faster than above
        M = np.zeros((nlen,nlen), dtype=dtype, order=order)
        for u,nbrdict in G.adjacency_iter():
            for v,d in nbrdict.items():
                try:
                    M[index[u],index[v]]=d.get(weight,1)
                except KeyError:
                    pass
        M = np.asmatrix(M)
    return M

def to_scipy_sparse_matrix(G, nodelist=None, dtype=None,
                           weight='weight', format='csr'):
    try:
        from scipy import sparse
    except ImportError:
        raise ImportError(\
          "to_scipy_sparse_matrix() requires scipy: http://scipy.org/ ")

    if nodelist is None:
        nodelist = G
    nlen = len(nodelist)
    if nlen == 0:
        raise NetworkXError("Graph has no nodes or edges")

    if len(nodelist) != len(set(nodelist)):
        msg = "Ambiguous ordering: `nodelist` contained duplicates."
        raise NetworkXError(msg)

    index = dict(zip(nodelist,range(nlen)))
    if G.number_of_edges() == 0:
        row,col,data=[],[],[]
    else:
        row,col,data=zip(*((index[u],index[v],d.get(weight,1))
                           for u,v,d in G.edges_iter(nodelist, data=True)
                           if u in index and v in index))
    if G.is_directed():
        M = sparse.coo_matrix((data,(row,col)),shape=(nlen,nlen), dtype=dtype)
    else:
        # symmetrize matrix
        M = sparse.coo_matrix((data+data,(row+col,col+row)),shape=(nlen,nlen),
                              dtype=dtype)
    try:
        return M.asformat(format)
    except AttributeError:
        raise NetworkXError("Unknown sparse matrix format: %s"%format)

class NetworkXException(Exception):
    """Base class for exceptions in NetworkX."""

class NetworkXError(NetworkXException):
    """Exception for a serious error in NetworkX"""