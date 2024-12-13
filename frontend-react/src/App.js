import React, { useEffect, useState } from "react";

function App() {
	const [message, setMessage] = useState("");

	useEffect(() => {
		fetch("http:/34.163.122.24/:8000/")
			.then((response) => response.json())
			.then((data) => setMessage(data.message));
	}, []);

	return (
		<div>
			<h1>React + FastAPIiiiiiiyyyyyyyyyy</h1>
			<p>{message}</p>
		</div>
	);
}

export default App;
