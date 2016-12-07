var svg;
var Y;

var canvas_width;
var canvas_height = 600;
var zoomListener;


function drawEmbedding() {
    $("#embed").empty();
    var div = d3.select("#embed");

    svg = div.append("svg") // svg is global
        .attr("width", canvas_width)
        .attr("height", canvas_height)
        .attr("style", "border:solid 1px");

    var g = svg.selectAll(".b")
      .data(data.samples)
      .enter().append("g")
      .attr("class", "u");

    g.append("text")
      .attr("text-anchor", "bottom")
      .attr("font-size", 11)
      .attr("font-family","'Ubuntu Mono', monospace")
      .attr("fill", "#111")
      .text(function(d) {
          if (d.is_test) return "t";
          if (d.is_neweq) return "n";
          if (d.is_val) return "v";
          return ""
       })
        .on("mouseover", point_mouseover)
        .on("mouseout", reset_highlight);

    g.append("circle")
        .attr("r", 5)
        .attr("fill", function(d){
             color = data.eq_classes[d.eq_class].color;
             col_name = d3.rgb(color[0] * 255, color[1] * 255, color[2] * 255);
             return col_name;})
        .attr("stroke", "black")
        .attr("opacity", .4)
        .on("mouseover", point_mouseover)
        .on("mouseout", reset_highlight);



    zoomListener = d3.behavior.zoom()
      .scaleExtent([0.0001, 500])
      .translate([canvas_width / 2, canvas_height / 2])
      .scale(ss)
      .on("zoom", zoomHandler);
    zoomListener(svg);

    updateEmbedding();
}


var line = d3.svg.line().interpolate("linear").x(function(d) {return d[0]*ss + tx}).y(function(d) {return d[1]*ss+ty});

function point_mouseover(point) {
    $("#Expression").text(point.expr);

    // Get all points in the same equivalence class
    other_points = [];
    for(var i=0; i < data.samples.length; i++) {
      if (data.samples[i].eq_class == point.eq_class) {
         var p = data.samples[i].xy;
         other_points.push([point.xy, p]);
      }
    }

    color = data.eq_classes[point.eq_class].color;
    var eq_class_color = d3.rgb(color[0] * 255, color[1] * 255, color[2] * 255);

    svg.selectAll(".line").data(other_points).enter()
        .append("path").attr("class", "line").attr("d", line)
        .attr("stroke", eq_class_color).attr("opacity", .4).attr("stroke-dasharray", ("10,3"));

    $("#EqClass").text(data.eq_classes[point.eq_class].name);
    $("#EqClassExtraInfo").text(" (" + other_points.length + " total in class)");
    $("#pointDataBrush").show();
}

function reset_highlight(d) {
    $("#pointDataBrush").hide();
    svg.selectAll('.line').remove();
}

function updateEmbedding() {
  svg.selectAll('.u')
    .attr("transform", function(d) {return "translate(" + ((d.xy[0]*ss + tx)) + "," + ((d.xy[1]*ss + ty)) + ")"; });
  svg.selectAll(".line").attr("d", line);
}

var tx;
var ty;
var ss=2;
function zoomHandler() {
  tx = d3.event.translate[0];
  ty = d3.event.translate[1];
  ss = d3.event.scale;
  updateEmbedding();
}

var data;
function load(json_path) {
	$("#embed").html('<h2><span class="glyphicon glyphicon-refresh glyphicon-refresh-animate"></span> Loading, please wait...</h2>');
    $.getJSON(json_path, function(json) {
        data = json
        drawEmbedding();
      });
}

$(window).load(function() {
    canvas_width = $(document).width() - 30;
    tx = canvas_width / 2;
    ty = canvas_height / 2;
    $("#pointDataBrush").hide();
    if (window.location.hash.length > 0) {
        load(window.location.hash.substr(1));
    }
 });


