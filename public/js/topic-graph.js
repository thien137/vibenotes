(function () {
  const data = window.__TOPIC_DATA__;
  if (!data || !data.notes || !data.edges) return;

  const base = ((data.pathPrefix || '').replace(/\/$/, '') || '');
  const notes = data.notes;
  const edges = data.edges;
  const rootNoteId = data.rootNoteId || (notes[0] && notes[0].id);

  const nodes = notes.map((n, i) => ({ ...n, index: i, x: 0, y: 0 }));

  const links = edges.filter(e => {
    const hasSource = notes.some(n => n.id === e.source);
    const hasTarget = notes.some(n => n.id === e.target);
    return hasSource && hasTarget;
  }).map(e => {
    const source = nodes.find(n => n.id === e.source);
    const target = nodes.find(n => n.id === e.target);
    return source && target ? { source, target } : null;
  }).filter(Boolean);

  const container = document.getElementById('graph-container');
  if (!container) return;

  const width = container.clientWidth || 800;
  const height = Math.min(600, window.innerHeight - 200);
  const nodeRadius = 64;
  const levelHeight = 155;
  const nodeSpacing = 165;

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', [0, 0, width, height])
    .attr('width', '100%')
    .attr('height', height);

  const g = svg.append('g');

  const defs = svg.append('defs');
  const linkColor = getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim() || '#6b6560';
  defs.append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '-0 -5 10 10')
    .attr('refX', 9)
    .attr('refY', 0)
    .attr('orient', 'auto')
    .attr('markerUnits', 'userSpaceOnUse')
    .attr('markerWidth', 10)
    .attr('markerHeight', 10)
    .append('path')
    .attr('d', 'M 0,-5 L 10,0 L 0,5')
    .attr('fill', linkColor);

  const linkG = g.append('g').attr('class', 'graph-links');
  const nodeG = g.append('g').attr('class', 'graph-nodes');

  let simulation = null;

  function assignLevels() {
    nodes.forEach(n => { n.level = -1; });
    const root = nodes.find(n => n.id === rootNoteId) || nodes[0];
    if (!root) return;
    root.level = 0;
    const queue = [root];
    while (queue.length) {
      const u = queue.shift();
      links.filter(l => l.source === u).forEach(l => {
        const v = l.target;
        if (v.level < 0) {
          v.level = u.level + 1;
          queue.push(v);
        }
      });
    }
    const maxLevel = Math.max(0, ...nodes.map(n => n.level));
    nodes.forEach(n => { if (n.level < 0) n.level = maxLevel + 1; });
  }

  function computeHierarchicalLayout() {
    assignLevels();
    const byLevel = {};
    nodes.forEach(n => {
      if (!byLevel[n.level]) byLevel[n.level] = [];
      byLevel[n.level].push(n);
    });
    const levels = Object.keys(byLevel).map(Number).sort((a, b) => a - b);
    levels.forEach((level, li) => {
      const levelNodes = byLevel[level];
      const totalW = levelNodes.length * nodeSpacing - (levelNodes.length - 1) * 24;
      let x = (width - totalW) / 2 + nodeSpacing / 2 - 10;
      levelNodes.forEach((n) => {
        n.x = x;
        n.y = 90 + li * levelHeight;
        x += nodeSpacing;
      });
    });
  }

  function linkPath(d) {
    const sx = d.source.x;
    const sy = d.source.y;
    const tx = d.target.x;
    const ty = d.target.y;
    const dx = tx - sx;
    const dy = ty - sy;
    const len = Math.hypot(dx, dy) || 0.001;
    const ux = dx / len;
    const uy = dy / len;
    const startX = sx + ux * (nodeRadius + 4);
    const startY = sy + uy * (nodeRadius + 4);
    const endX = tx - ux * (nodeRadius + 4);
    const endY = ty - uy * (nodeRadius + 4);
    return `M ${startX},${startY} L ${endX},${endY}`;
  }

  function ticked() {
    linkG.selectAll('path').attr('d', linkPath);
    nodeG.selectAll('g').attr('transform', d => `translate(${d.x},${d.y})`);
  }

  const link = linkG.selectAll('path')
    .data(links)
    .join('path')
    .attr('fill', 'none')
    .attr('stroke', linkColor)
    .attr('stroke-opacity', 0.6)
    .attr('stroke-width', 2)
    .attr('marker-end', 'url(#arrowhead)');

  const bubbleVars = ['--bubble-1', '--bubble-2', '--bubble-3', '--bubble-4', '--bubble-5'];
  const fallbackColors = ['#ffb5a7', '#a8e6cf', '#c9b1bd', '#b8d4e8', '#ffd6ba'];
  const getNodeColor = (i) => {
    const val = getComputedStyle(document.documentElement).getPropertyValue(bubbleVars[i % 5]).trim();
    return val || fallbackColors[i % 5];
  };
  const strokeColor = getComputedStyle(document.documentElement).getPropertyValue('--graph-node-text').trim()
    || getComputedStyle(document.documentElement).getPropertyValue('--text-dark').trim()
    || '#2d2a26';

  const node = nodeG.selectAll('g')
    .data(nodes)
    .join('g')
    .attr('cursor', 'pointer')
    .call(d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended));

  node.append('circle')
    .attr('r', nodeRadius)
    .attr('fill', (d, i) => getNodeColor(i))
    .attr('stroke', strokeColor)
    .attr('stroke-width', 1.5);

  node.append('text')
    .attr('text-anchor', 'middle')
    .attr('font-size', 13)
    .attr('font-family', 'Quicksand, sans-serif')
    .attr('fill', strokeColor)
    .attr('pointer-events', 'none')
    .style('user-select', 'none')
    .each(function (d) {
      const t = d3.select(this);
      const words = (d.title || '').split(/\s+/);
      t.text(null);
      if (words.length === 1) {
        t.append('tspan').attr('x', 0).attr('dy', '0.35em').text(d.title);
      } else if (words.length === 2) {
        t.append('tspan').attr('x', 0).attr('dy', '-0.35em').text(words[0]);
        t.append('tspan').attr('x', 0).attr('dy', '1.2em').text(words[1]);
      } else {
        const mid = Math.ceil(words.length / 2);
        t.append('tspan').attr('x', 0).attr('dy', '-0.35em').text(words.slice(0, mid).join(' '));
        t.append('tspan').attr('x', 0).attr('dy', '1.2em').text(words.slice(mid).join(' '));
      }
    });

  const tooltip = d3.select('body').append('div')
    .attr('class', 'graph-tooltip')
    .style('opacity', 0)
    .style('pointer-events', 'none');

  node.on('mouseenter', function (event, d) {
    d3.select(this).select('circle').attr('stroke-width', 2.5);
    const imgPath = d.image ? (base + '/' + d.image.replace(/^\//, '')) : '';
    tooltip
      .style('opacity', 1)
      .html(`
        <div class="tooltip-content">
          <div class="tooltip-title">${d.title || ''}</div>
          ${imgPath ? `<img src="${imgPath}" alt="" class="tooltip-img" />` : ''}
          <p class="tooltip-summary">${d.summary || ''}</p>
          <span class="tooltip-click">Click to open note</span>
        </div>
      `)
      .style('left', (event.pageX + 12) + 'px')
      .style('top', (event.pageY + 12) + 'px');
  })
  .on('mouseleave', function () {
    d3.select(this).select('circle').attr('stroke-width', 1.5);
    tooltip.style('opacity', 0);
  })
  .on('mousemove', event => {
    tooltip.style('left', (event.pageX + 12) + 'px').style('top', (event.pageY + 12) + 'px');
  })
  .on('click', (event, d) => {
    event.preventDefault();
    window.location.href = base + '/notes/' + d.id + '/';
  });

  const organizeBtn = document.getElementById('organize-btn');
  if (organizeBtn) {
    organizeBtn.addEventListener('click', () => {
      computeHierarchicalLayout();
      if (simulation) simulation.alpha(0.3).restart();
      ticked();
    });
  }

  svg.call(d3.zoom()
    .scaleExtent([0.25, 4])
    .on('zoom', (event) => g.attr('transform', event.transform)));

  computeHierarchicalLayout();
  simulation = d3.forceSimulation(nodes)
    .force('x', d3.forceX(d => d.x).strength(0.04))
    .force('y', d3.forceY(d => d.y).strength(0.04))
    .force('collision', d3.forceCollide().radius(nodeRadius + 25))
    .on('tick', ticked);

  function dragstarted(event) {
    event.sourceEvent.stopPropagation();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
    if (!event.active && simulation) simulation.alphaTarget(0.3).restart();
  }

  function dragged(event) {
    event.subject.x = event.x;
    event.subject.y = event.y;
    event.subject.fx = event.x;
    event.subject.fy = event.y;
    if (simulation) simulation.alpha(0.1).restart();
    ticked();
  }

  function dragended(event) {
    event.subject.fx = null;
    event.subject.fy = null;
    event.subject.x = event.x;
    event.subject.y = event.y;
    if (!event.active && simulation) simulation.alphaTarget(0);
  }
})();
