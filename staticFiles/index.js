const clock = document.getElementById("clock");

setInterval(() => {
    fetch("{{ url_for('time_feed') }}")
        .then(response => {
            response.text().then(t => {
                clock.innerHTML = t
            })
        });
}, 2000);
