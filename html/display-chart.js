function plotData(data, divId) {
    var margin = {
        top: 30,
        right: 20,
        bottom: 30,
        left: 50
    };
    var width = 600 - margin.left - margin.right;
    var height = 270 - margin.top - margin.bottom;

    var x = d3.scaleTime().range([0, width]);
    var y = d3.scaleLinear().range([height, 0]);

    var xAxis = d3.axisBottom(x).ticks(5);

    var yAxis = d3.axisLeft(y).ticks(5);

    var valueline = d3.line()
        .x(function (d) {
            return x(d.date);
        })
        .y(function (d) {
            return y(d.y);
        });
    var divSelector = "#" + divId;
    d3.select(divSelector).select("svg").remove();
    var svg = d3.select(divSelector)
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // Scale the range of the data
    x.domain(d3.extent(data, function (d) {
        return d.date;
        }));
    y.domain([0, d3.max(data, function (d) {
        return d.y;
        })]);

    svg.append("path") // Add the valueline path.
    .attr("d", valueline(data));

    svg.append("g") // Add the X Axis
    .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    svg.append("g") // Add the Y Axis
    .attr("class", "y axis")
        .call(yAxis);
}
